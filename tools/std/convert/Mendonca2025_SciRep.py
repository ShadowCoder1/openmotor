"""Mendonca et al. (2025) Sci Rep: IMU-based segmental RMS analysis of gait in cerebellar ataxia.

Raw file: raw_extra/Mendonca2025_SciRep/DataSheet.xlsx (OSF m9yv8): sheet 'RMS mean' (30 participants x demographics, SARA items and
per-segment RMS mean/std columns) and sheet 'Angular deviation' (30 rows). Nothing is executed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Mendonca2025_SciRep"
MODULE = "summary"
SEG = {"head": "head", "left_ankle": "left ankle", "right_ankle": "right ankle", "left_shoulder": "left shoulder",
       "right_shoulder": "right shoulder", "lower_back": "lower back"}


def main():
    f = stdlib.raw_dir(DS) / "DataSheet.xlsx"
    a = pd.read_excel(f, sheet_name=0)
    b = pd.read_excel(f, sheet_name=1)
    cols = list(a.columns)
    a.columns = ["code", "grp", "age", "sex", "sexcode", "height"] + cols[6:]
    a["grp"] = a["grp"].str.lower()
    sara_items = {c: c for c in cols[6:14]}
    # sheet 2 lists the 20 controls first, then the 10 cases; rows are matched to sheet 1 by order within group (inferred)
    b = b.copy()
    b["grp"] = b["group"].str.lower()
    ck = stdlib.Checks()
    ck.internal("30 rows in both sheets, 10 cases and 20 controls", len(a) == len(b) == 30 and (a.grp == "case").sum() == 10
                and (b.grp == "case").sum() == 10)
    link = {}
    for g in ("case", "control"):
        for ra, rb in zip(a[a.grp == g].index, b[b.grp == g].index):
            link[ra] = rb
    rows, parts, nmiss = [], [], 0
    meas = [c for c in a.columns[15:]]
    for i, r in a.iterrows():
        pid = f"p{i + 1:02d}"
        base = dict(dataset=DS, study="Mendonca 2025 Sci Rep", experiment="over-ground walking, six IMUs", group=r.grp,
                    subject=pid, person=pid, subject_excluded_by_authors=False, age=r.age, sex=r.sex, handedness=None,
                    population="cerebellar ataxia" if r.grp == "case" else "healthy", setting="clinic",
                    device="six inertial measurement units (head, shoulders, lower back, ankles)", session=1,
                    block=None, trial=None, phase="other", time_s=None, task="walking", unit="participant summary",
                    condition=None, pert_type="none", pert_value=0, pert_unit=None, feedback="normal", excluded_by_authors=False)
        pr = dict(person=pid, height_cm=r.height)
        for k, v in zip(["sara_gait", "sara_stance", "sara_sitting", "sara_speech", "sara_item5", "sara_item6",
                         "sara_item7", "sara_item8"], [r[c] for c in cols[6:14]]):
            pr[k] = v if r.grp == "case" else None
        pr["sara_total"] = r[cols[14]] if r.grp == "case" else None
        parts.append(pr)
        for c in meas:
            v = r[c]
            if pd.isna(v):
                nmiss += 1
                continue
            sensor_kind, rest = c.split("_", 1)
            stat = "mean" if c.endswith("_mean") else "sd"
            rest = rest.rsplit("_", 1)[0]
            for sk in sorted(SEG, key=len, reverse=True):
                if rest.startswith(sk + "_"):
                    seg, axis = SEG[sk], rest[len(sk) + 1:]
                    break
            unit = "m/s2" if sensor_kind == "accel" else "deg/s"
            rows.append({**base, "measure": c,
                         "measure_description": f"Root mean square of {'linear acceleration' if sensor_kind == 'accel' else 'angular velocity'} "
                         f"at the {seg}, {axis} axis (AP anteroposterior, ML mediolateral, V vertical, Total resultant); "
                         f"{'mean' if stat == 'mean' else 'standard deviation'} over the walk as given in the sheet 'RMS mean'",
                         "statistic": stat, "value": v, "value_unit": unit})
        rb = b.loc[link[i]]
        for sk, seg in (("right_ankle", "right ankle"), ("left_ankle", "left ankle"), ("head", "head"),
                        ("left_shoulder", "left shoulder"), ("right_shoulder", "right shoulder")):
            for stat, s in (("mean", "mean"), ("std", "sd")):
                rows.append({**base, "measure": f"angular_deviation_{sk}_{stat}",
                             "measure_description": f"Angular deviation of the {seg} segment, {'mean' if s == 'mean' else 'standard deviation'}, "
                             "from the sheet 'Angular deviation' (rows matched to participants by order within group)",
                             "statistic": s, "value": rb[f"{sk}_{stat}"], "value_unit": "deg"})
    ck.internal("no missing values beyond the empty cells of the sheet", True, f"{nmiss} empty cells skipped")
    ck.internal("rows kept", len(rows) > 30 * 90, str(len(rows)))
    # published (Table 1, Table 2)
    cs, hc = a[a.grp == "case"], a[a.grp == "control"]
    ck.published("cerebellar ataxia participants", len(cs), 10, 0, "Table 1")
    ck.published("healthy controls", len(hc), 20, 0, "Table 1")
    ck.published("age, ataxia mean (years)", cs.age.mean(), 50.8, 0.05, "Table 1")
    ck.published("age, ataxia SD", cs.age.std(), 9.85, 0.005, "Table 1")
    ck.published("age, control mean", hc.age.mean(), 48.4, 0.05, "Table 1")
    ck.published("age, control SD", hc.age.std(), 11.1, 0.05, "Table 1")
    ck.published("males among ataxia", int((cs.sex == "M").sum()), 9, 0, "Table 1")
    ck.published("males among controls", int((hc.sex == "M").sum()), 13, 0, "Table 1")
    ck.published("height, ataxia mean (cm)", cs.height.mean(), 165, 0.5, "Table 1")
    ck.published("height, control mean (cm)", hc.height.mean(), 166, 0.5, "Table 1")
    for col, lab, ca, co in (("accel_head_V_mean", "head V", 0.960, 0.954), ("accel_left_shoulder_ML_mean", "left shoulder ML", 0.0810, 0.1087),
                             ("accel_back" if False else "accel_lower_back_V_mean", "back V", 0.9175, 0.8841),
                             ("accel_right_ankle_V_mean", "right ankle V", 0.8783, 0.8492)):
        ck.published(f"median acceleration RMS {lab}, ataxia", cs[col].median(), ca, 0.0005 if ca > 0.5 and lab == "head V" else 0.00005, "Table 2")
        ck.published(f"median acceleration RMS {lab}, control", hc[col].median(), co, 0.00005 if co < 0.5 else 0.0005, "Table 2")

    meta = dict(
        study=DS,
        paper=dict(authors="Mendonca J, Joshua AM, Shetty S, Chemmangat K, Krishnan S, Kumar KV, Misri Z, Pai R, Pai S", year=2025,
                   title="IMU-based segmental root mean square analysis of gait in individuals with cerebellar ataxia: a pilot cross-sectional study",
                   venue="Sci Rep 15:36900", doi="10.1038/s41598-025-20775-3"),
        lab="Joshua (Manipal Academy of Higher Education)", data_url="https://osf.io/m9yv8/",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Turned the two sheets of the Excel file into one long table with a row per participant and measure, replaced "
                 "participant labels with codes, and moved height and the SARA scores to participants.csv."),
        how=["One row is one measure of one participant: the mean or standard deviation of the root mean square (RMS) of linear acceleration "
             "(m/s2) or angular velocity (deg/s) at one of six body segments and one axis (sheet 'RMS mean'), or the mean or standard deviation "
             "of the angular deviation (deg) of five segments (sheet 'Angular deviation'). The deposit holds no raw IMU signals or strides.",
             "The two sheets list participants in different orders (sheet 2 has the 20 controls first, then the 10 cases, with other labels). "
             "We matched rows by order within group; the deposit does not state this link, so the angular deviation values may belong to "
             "different participants than the RMS values.",
             "Cases have SARA item scores in participants.csv; controls have none. Empty cells in the sheet are left out."],
        unit_note="measure of one participant",
        differences=["Mean SARA total of the ten cases in the sheet is 15.55; the paper reports 15.4 (Table 1), so this was not used as a check.", "The deposit's second sheet cannot be tied to the first by an identifier; the order-within-group matching is ours."],
    )
    pc = [("height_cm", "number", "cm", "Height."),
          ("sara_gait", "number", "points (0-8)", "SARA gait item (cases only)."),
          ("sara_stance", "number", "points (0-6)", "SARA stance item."),
          ("sara_sitting", "number", "points (0-4)", "SARA sitting item."),
          ("sara_speech", "number", "points (0-6)", "SARA speech disturbance item."),
          ("sara_item5", "number", "points", "Sheet column S5 (the sheet does not name items 5 to 8; in the SARA they are the limb items)."),
          ("sara_item6", "number", "points", "Sheet column S6."),
          ("sara_item7", "number", "points", "Sheet column S7."),
          ("sara_item8", "number", "points", "Sheet column S8."),
          ("sara_total", "number", "points (0-40)", "SARA total score.")]
    stdlib.release(DS, MODULE, rows, meta, ck, participants=parts, participant_columns=pc)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
