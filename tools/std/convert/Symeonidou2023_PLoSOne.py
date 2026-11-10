"""Symeonidou, Esposito, Reyes & Ferris (2023) PLoS ONE: practice walking on a treadmill-mounted balance beam, with or without
intermittent visual occlusions, and transfer to other balance tasks, 20 young adults (10 per group), tested before and after.

Raw file: raw_extra/Symeonidou2023_PLoSOne/symeonidou.xlsx (Figshare 23258048, CC BY 4.0): one sheet, rows = participant x group x
Pre/Post (2 header rows then 40 rows), 29 measure columns in four blocks (Tables 1-3 and Figure 3 of the paper).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Symeonidou2023_PLoSOne"
MODULE = "summary"
SECTIONS = ["Table 1: On-beam balancing", "Table 2: Single-leg stance", "Table 3: Tandem stance", "Figure 3: Walking margin of stability"]
# Table 1 of the paper: mean and SD per measure for (Unperturbed Pre, Unperturbed Post, Occlusion Pre, Occlusion Post), 9 participants per group
T1 = {
    "Step-Offs/min": ([15.50, 11.97, 9.74, 2.43], [5.15, 5.90, 4.67, 2.09]),
    "Sac Standard Deviation ML": ([3.79, 3.40, 3.32, 2.32], [1.39, 1.76, 1.66, 0.88]),
    "Sac Standard Deviation Vert.": ([1.28, 1.41, 1.11, 1.17], [0.27, 0.81, 0.21, 0.39]),
    "Sac Standard Deviation AP": ([6.72, 5.51, 5.72, 5.10], [1.77, 1.42, 1.24, 1.19]),
    "Sac Max Displacement ML": ([13.34, 13.32, 12.61, 9.17], [2.73, 5.39, 4.12, 4.03]),
    "Sac Max Displacement Vert": ([4.49, 4.84, 4.54, 4.46], [1.25, 2.05, 1.04, 1.76]),
    "Sac Max Displacement AP": ([19.27, 18.03, 17.09, 14.76], [5.55, 4.91, 4.16, 3.42]),
    "Sac Max Velocity ML": ([45.24, 45.89, 39.76, 34.60], [7.09, 11.36, 6.85, 12.02]),
    "Sac Max Velocity Vert.": ([30.69, 35.88, 18.06, 21.58], [7.25, 15.61, 6.38, 10.38]),
    "Sac Max Velocity AP": ([34.51, 37.07, 32.49, 33.57], [7.00, 11.96, 8.21, 10.92]),
    "Norm Ankle-Sacral Distance": ([100.71, 99.00, 100.26, 99.59], [1.08, 1.33, 0.95, 1.07]),
}


def unit_stat(sec, m):
    ml = m.lower()
    if sec == 0:
        if "step-offs" in ml:
            return "step-offs per minute", "value"
        if "norm" in ml:
            return "normalized, unitless (about 100)", "value"
        if "velocity" in ml:
            return "cm/s", "value"
        return "cm", "sd" if "standard deviation" in ml else "value"
    if sec in (1, 2):
        if "velocity" in ml:
            return "cm/s", "value"
        if "area" in ml:
            return "cm^2", "value"
        return "cm", "sd" if ml.startswith("sd") else "value"
    return "not stated by the authors", "sd" if "variability" in ml else "mean"


def main():
    x = pd.read_excel(stdlib.raw_dir(DS) / "symeonidou.xlsx", header=None)
    names = [str(c).strip() for c in x.iloc[1, 3:]]
    starts = [i for i, v in enumerate(x.iloc[0, 3:]) if isinstance(v, str)]
    assert len(starts) == 4
    sec_of = [max(k for k, s in enumerate(starts) if s <= i) for i in range(len(names))]
    d = x.iloc[2:, :].reset_index(drop=True)
    d.columns = ["pid", "grp", "tt"] + [f"c{i}" for i in range(len(names))]  # measure names repeat across blocks
    ck = stdlib.Checks()
    rows = []
    for r in d.itertuples(index=False):
        rec = dict(zip(d.columns, r))
        pre = rec["tt"] == "Pre"
        grp = "visual occlusions" if rec["grp"] == "Visual Occlusions" else "unperturbed vision"
        code = f"{grp[:3].upper()}{int(rec['pid']):02d}"
        base = dict(dataset=DS, study="Symeonidou 2023 PLoS ONE", experiment="balance beam training and transfer",
                    group=grp, subject=code, person=code, subject_excluded_by_authors=False, age=None, sex=None, handedness=None,
                    population="healthy", setting="lab", device="treadmill-mounted balance beam, motion capture, force plate",
                    session=1, block=rec["tt"], trial=None, phase="baseline" if pre else "test", time_s=None, task="balance and posture",
                    unit="participant summary", condition=None, pert_type="none", pert_value=0, pert_unit=None,
                    feedback="normal", excluded_by_authors=False)
        for i, (m, sec) in enumerate(zip(names, sec_of)):
            v = rec[f"c{i}"]
            if pd.isna(v):
                continue
            unit, stat = unit_stat(sec, m)
            rows.append({**base, "condition": SECTIONS[sec],
                         "measure": m, "measure_description": f"{m} ({SECTIONS[sec]}), {rec['tt'].lower()}-training test, as named in the deposit",
                         "statistic": stat, "value": v, "value_unit": unit})
    ck.internal("40 lines (10 participants x 2 groups x Pre/Post)", len(d) == 40 and d.groupby(["grp", "tt"]).size().eq(10).all())
    ck.internal("rows = non-empty cells", len(rows) == int(d[[f"c{i}" for i in range(len(names))]].notna().sum().sum()), str(len(rows)))
    ck.internal("2 participants (one per group) have completely empty lines, so 18 have rows", len({r["subject"] for r in rows}) == 18)
    ck.published("participants", d.pid.groupby(d.grp).nunique().sum(), 20, 0, "Methods, Participants (twenty healthy young participants)")
    ck.published("participants per group", d[d.grp == "Visual Occlusions"].pid.nunique(), 10, 0, "Methods, Beam training (4F+6M and 5F+5M)")
    cells = [("Unperturbed Vision", "Pre"), ("Unperturbed Vision", "Post"), ("Visual Occlusions", "Pre"), ("Visual Occlusions", "Post")]
    for m, (means, sds) in T1.items():
        for (g, t), mu, sd in zip(cells, means, sds):
            s = pd.to_numeric(d[(d.grp == g) & (d.tt == t)][f"c{names.index(m)}"], errors="coerce").dropna()
            ck.published(f"{m}: {g} {t} mean (n={len(s)})", s.mean(), mu, 0.005, "Table 1")
            ck.published(f"{m}: {g} {t} SD", s.std(), sd, 0.005, "Table 1")
    meta = dict(
        study=DS,
        paper=dict(authors="Symeonidou ER, Esposito NM, Reyes RD, Ferris DP", year=2023,
                   title="Practice walking on a treadmill-mounted balance beam modifies beam walking sacral movement and alters performance in other balance tasks",
                   venue="PLoS ONE 18(6):e0283310", doi="10.1371/journal.pone.0283310"),
        lab="Ferris (University of Florida)", data_url="https://doi.org/10.6084/m9.figshare.23258048",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Turned the wide table (one line per participant, group and test) into a long table with one row per participant, "
                 "test and measure (empty cells are not written), stripped leading spaces from measure names, and gave participants "
                 "codes made of group and number (the deposit numbers 1-10 inside each group)."),
        how=["One row is one measure of one participant before (Pre, phase baseline) or after (Post, phase test) 30 minutes of treadmill "
             "balance-beam walking practice, in one of two groups (practice with intermittent visual occlusions, or with unperturbed vision). "
             "condition names the block of measures (on-beam balancing, single-leg stance, tandem stance, walking margin of stability). "
             "The deposit shares participant-level summaries only.",
             "Units follow the paper's table captions: cm, cm/s, cm^2, step-offs per minute, and a normalized ankle-sacral distance; the units of "
             "the margin-of-stability values are not stated. statistic is sd for the measures that are themselves standard deviations "
             "or variabilities, mean for the mean margins of stability, and value otherwise (maxima, sway, areas, counts). Which of these "
             "descriptions applies was read from the measure names. Which participant is which across the Pre and Post lines is by the "
             "deposit's participant number inside each group.",
             "Missing cells (blank in the deposit) were not written; in the single-leg and tandem stance blocks a few participants lack values."],
        unit_note="measure of one participant before or after training",
        differences=["Two of the 20 participants (one per group) have completely blank Pre and Post lines in the deposit, so no rows exist for them (18 participants in the release, 9 per group, which matches the 9 per group of the paper's Table 1).", "Table 1 of the paper is computed on 9 participants per group (the caption says so); the deposit lists 10 per group, one of whom is blank, and all Table 1 means and SDs reproduce on the 9 with values."])
    stdlib.release(DS, MODULE, rows, meta, ck)


if __name__ == "__main__":
    main()
