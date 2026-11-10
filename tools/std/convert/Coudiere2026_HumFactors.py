"""Coudiere, Morin, Bernier & Danion (2025), Hum Factors 68(1):92-108: unimanual and bimanual joystick tracking (2D target per
hand), 28 right-handed adults, gaze free / fixed / INVERTED (authors' label), 10 trials per condition.

Raw file raw_extra/Coudiere2026_HumFactors/DUAL.xlsx (OSF hv6cd, no license stated; original DUAL-TRACK-1Page.xlsx), one sheet
with stacked tables. Used here: the participant table (rows 4-31) and the six per-trial tables with a title row, a gaze-label row,
a Trial row and 28 participant rows: right/left hand unimanual tracking error (cm), right/left hand unimanual lag (ms),
right/left hand bimanual tracking error (cm). Not released (derived by the authors from these): MEAN rows, OVERALL columns,
right+left averages, error increase, asymmetry ratio, and the gaze-position summary tables.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Coudiere2026_HumFactors"
SRC = "Coudiere et al. 2025, Human Factors"
N = 28
TABLES = [(37, "RIGHT HAND UNIMANUAL", "tracking_error_right_hand", "cm", "unimanual", "right"),
          (72, "LEFT HAND UNIMANUAL", "tracking_error_left_hand", "cm", "unimanual", "left"),
          (107, "RIGHT HAND UNIMANUAL", "lag_right_hand", "ms", "unimanual", "right"),
          (142, "LEFT HAND UNIMANUAL", "lag_left_hand", "ms", "unimanual", "left"),
          (215, "RIGHT HAND BIMANUAL", "tracking_error_right_hand", "cm", "bimanual", "right"),
          (250, "LEFT HAND BIMANUAL", "tracking_error_left_hand", "cm", "bimanual", "left")]
DESC = {"cm": "Mean cursor-target distance of one trial for the {h} hand, in cm.",
        "ms": "Lag between cursor and target for one trial for the {h} hand, in ms."}


def main():
    d = pd.read_excel(stdlib.raw_dir(ID) / "DUAL.xlsx", header=None)
    info = d.iloc[4:4 + N, :4]
    assert list(info[0]) == [f"S{i}" for i in range(1, N + 1)]
    rows = []
    for t0, prefix, name, unit, mode, hand in TABLES:
        assert str(d.iloc[t0, 0]).upper().startswith(prefix), (t0, d.iloc[t0, 0])
        assert (unit == "cm") == ("(cm" in str(d.iloc[t0, 0]).lower() or "ERR" in str(d.iloc[t0, 0]).upper())
        for p in range(N):
            assert d.iloc[t0 + 4 + p, 0] == f"S{p + 1}"
            k = 0
            for c in range(1, 31):
                gaze = str(d.iloc[t0 + 2, c if c % 2 else c - 1]).title()
                trial = int(str(d.iloc[t0 + 3, c]).replace("Trial", ""))
                v = d.iloc[t0 + 4 + p, c]
                k += 1
                rows.append({"dataset": ID, "study": SRC, "experiment": mode, "group": "", "subject": f"S{p + 1}",
                             "subject_excluded_by_authors": False, "age": float(info.iloc[p, 2]), "sex": info.iloc[p, 1],
                             "handedness": "right", "population": "healthy", "setting": "lab",
                             "device": "two joysticks controlling cursors, eye tracker", "session": 1,
                             "block": f"{gaze}; trial {trial}", "trial": k, "phase": "test", "time_s": "", "task": "tracking",
                             "unit": "trial", "condition": f"{mode}; {hand} hand; gaze {gaze.lower()}", "pert_type": "none",
                             "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
                             "session_context": "", "measure": name, "measure_description": DESC[unit].format(h=hand),
                             "statistic": "value", "value": None if pd.isna(v) else float(v), "value_unit": unit, "n_trials": 1})
    df = pd.DataFrame(rows)
    parts = [{"person": f"S{p + 1}", "laterality_quotient": float(info.iloc[p, 3])} for p in range(N)]
    ck = stdlib.Checks()
    ab = "Abstract (Europe PMC record of the paper)"
    ck.published("participants", df.subject.nunique(), 28, 0, ab)
    ages = info[2].astype(float)
    ck.published("mean age (years, printed as 22)", ages.mean(), 22, 0.5, ab)
    ck.internal("rows = 6 tables x 28 persons x 30 trials", len(df) == 6 * N * 30, str(len(df)))
    nmiss = int(df.value.isna().sum())
    ck.internal("few empty cells (left empty as in the deposit)", nmiss < 20, str(nmiss))
    ck.internal("tracking errors positive, lags finite", (df[df.value_unit == "cm"].value > 0).all())
    for c in ck.items:
        print(("ok " if c["ok"] else "FAILED "), c["what"], c.get("computed", ""))
    s = next(z for z in json.load(open(stdlib.HERE / "source/extra_studies.json")) if z["id"] == ID)
    pp = s["papers"][0]
    meta = dict(study=ID, paper={k: pp[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
                license="none stated", license_url="",
                changes="Stacked six per-trial tables of the Excel sheet into long form, added sex, age from the participant table, left out MEAN/OVERALL columns and derived tables, replaced participant labels with codes. Values unchanged.",
                how=["One row is one trial of one participant for one hand: tracking error (mean cursor-target distance, cm) or lag between cursor and target (ms), unimanual or bimanual, with gaze free, fixed or INVERTED (the authors' label; its meaning is not documented in the deposit). Each condition has 10 trials; trial counts 1-30 follow the authors' column order (free, fixed, inverted), and the block column keeps the authors' trial label.",
                     "Lag is given by the deposit for unimanual tracking only. Participants were right-handed per the paper (handedness filled from the abstract); participants.csv holds the laterality quotient. The authors' right+left averages, error-increase and asymmetry-ratio tables and gaze-position summaries are derived and not released."],
                unit_note="trial of one participant, one hand and one measure (long form)",
                differences=[f"{nmiss} cells are empty in the deposit and stay empty here.", "The sheet contains derived tables and per-participant means that are not released; the paper's statistics were not re-computed. Checks use the abstract's sample size and mean age."])
    return stdlib.release(ID, "summary", df, meta, ck, participants=parts,
                          participant_columns=[("laterality_quotient", "number", "", "Laterality quotient from the participant table (scale as in the deposit, not documented there).")])


if __name__ == "__main__":
    main()
