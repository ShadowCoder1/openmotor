"""Shida et al. (2023) Front Neurosci 17:992585: overground walking of 26 people with Parkinson's disease, ON and OFF
medication, full-body kinematics and kinetics (Figshare 10.6084/m9.figshare.14896881, CC BY 4.0).

Raw files (raw_extra/Shida2023_FrontNeurosci/): Gait_cycles.zip (116 MB: per participant and medication state, 24 Excel 97
workbooks with a .csv extension; the *_spatiotemporal.csv workbooks hold one row per walking pass and are converted here;
the others hold time-normalized (101 points) mean and SD curves of joint angles, ground reaction forces and the centre of
mass and are not converted) and PDGinfo.xlsx (anthropometrics and clinical scores). The 1.3 GB C3Dfiles.zip (raw marker and
force-plate trials) is not converted.
Run from the site folder: python3 -I tools/std/convert/Shida2023_FrontNeurosci.py"""
import io
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

warnings.filterwarnings("ignore")
DS = "Shida2023_FrontNeurosci"
raw = stdlib.raw_dir(DS)
ck = stdlib.Checks()
DEVICE = ("20 m overground walkway, barefoot, self-selected speed; 12-camera Motion Analysis Raptor-4 motion capture "
          "(44 markers, 150 Hz); five embedded force plates (3 AMTI, 2 Kistler)")

# ---- participant information
info = pd.read_excel(raw / "PDGinfo.xlsx", engine="openpyxl")
info.columns = [str(c).strip() for c in info.columns]
col = {re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_"): c for c in info.columns}


def C(key):
    hits = [v for k, v in col.items() if k == key]
    assert hits, (key, list(col))
    return info[hits[0]]


ids = info["ID"].astype(str).str.strip()
assert ids.is_unique and len(ids) == 26, len(ids)
age, sex = dict(zip(ids, C("age"))), dict(zip(ids, C("gender")))
hand_raw = dict(zip(ids, C("handedness").astype(str)))


def hand(v):
    v = str(v).strip().lower()
    return "right" if v.startswith("r") else "left" if v.startswith("l") else "ambi" if v[:1] in ("a", "b") else ""


# ---- trials: one row per walking pass
SPEC = {  # workbook column -> module column
    "Left_Stance_Time_Mean": "stance_time_left_s", "Left_Swing_Time_Mean": "swing_time_left_s",
    "Left_Step_Length_Mean": "step_length_left_m", "Left_Steps_Per_Minute_Mean": "steps_per_min_left",
    "Left_Stride_Length_Mean": "stride_length_left_m", "Left_Strides_Per_Minute_Mean": "strides_per_min_left",
    "Right_Stance_Time_Mean": "stance_time_right_s", "Right_Swing_Time_Mean": "swing_time_right_s",
    "Right_Step_Length_Mean": "step_length_right_m", "Right_Steps_Per_Minute_Mean": "steps_per_min_right",
    "Right_Stride_Length_Mean": "stride_length_right_m", "Right_Strides_Per_Minute_Mean": "strides_per_min_right",
    "Speed": "walking_speed_mps", "Stride_Length_Mean": "stride_length_m", "Stride_Width_Mean": "stride_width_m",
    "Cycle_Time_Mean": "stride_time_s", "Double_Limb_Support_Time_Ave": "double_support_s",
    "Right_Initial_Double_Limb_Support_Time_Mean": "double_support_right_initial_s",
    "Right_Terminal_Double_Limb_Support_Time_Mean": "double_support_right_terminal_s"}
z = zipfile.ZipFile(raw / "Gait_cycles.zip")
files = sorted(n for n in z.namelist() if n.endswith("_spatiotemporal.csv"))
rows, sessions = [], []
for fn in files:
    m = re.search(r"(SUB\d+)_(on|off)_walk_spatiotemporal", fn)
    pid, state = m.group(1), m.group(2).upper()
    t = pd.read_excel(io.BytesIO(z.read(fn)), engine="xlrd", header=0)
    t.columns = [str(c).strip() for c in t.columns]
    assert set(SPEC) <= set(t.columns), (fn, set(SPEC) - set(t.columns))
    sessions.append((pid, state, len(t)))
    for i, r in enumerate(t.itertuples(index=False), start=1):
        rd = dict(zip(t.columns, r))
        d = dict(dataset=DS, study=DS, experiment="", group="", subject=f"{pid}_{state}", person=pid,
                 subject_excluded_by_authors=False, age=float(age[pid]), sex=sex[pid], handedness=hand(hand_raw[pid]),
                 population="Parkinson's disease", setting="lab", device=DEVICE, session=1, block=state, trial=i,
                 phase="test", task="walking", unit="trial", condition=f"medication {state}", pert_type="none",
                 pert_value=0.0, pert_unit="", feedback="normal", excluded_by_authors=False, fast_leg="unknown",
                 walking_context="overground", surface_incline_deg=0.0)
        for k, v in SPEC.items():
            d[v] = rd[k]
        rows.append(d)
df = pd.DataFrame(rows).reindex(columns=[c[1] for c in stdlib.columns_for("walking") if c[1] != "person"] + ["person"])
df = df[[c[1] for c in stdlib.columns_for("walking")]]

# ---- participants.csv: numeric and categorical items of PDGinfo.xlsx (free-text history items are left out)
FREE = {"initial_symptoms", "is_there_a_family_history_of_pd_who", "do_you_feel_improvement_after_using_the_antiparkinsonian_medicine",
        "have_you_ever_had_any_type_of_surgery_which", "any_rehabilitation_or_physical_activity", "other_disease_cardiovascular_bone_etc",
        "id", "gender", "age", "handedness"}
pcols, pdata = [], {}
for k, orig in col.items():
    if k in FREE:
        continue
    name = k
    s = info[orig]
    num = pd.to_numeric(s, errors="coerce")
    is_num = num.notna().sum() >= 0.5 * s.notna().sum() and s.notna().sum() > 0
    pcols.append((name, "number" if is_num else "text", "", f"'{orig}' in the deposit's PDGinfo.xlsx (clinical and anthropometric "
                  "information collected for each participant; ON and OFF are the medication states)."))
    pdata[name] = num if is_num else s.astype(str).where(s.notna(), "")
part = pd.DataFrame(pdata)
part.insert(0, "person", ids.values)
part["person"] = part["person"].astype(str)
part_people = [dict(zip(part.columns, r)) for r in part.itertuples(index=False)]
part_people = [p for p in part_people if p["person"] in set(df.person)]
for p in part_people:
    for k, v in list(p.items()):
        if isinstance(v, float) and np.isnan(v):
            p[k] = ""

# ---- checks
ck.internal("26 participants, all with walking data; 48 session files (25 ON, 23 OFF)",
            df.person.nunique() == 26 and len(sessions) == 48 and sum(s[1] == "ON" for s in sessions) == 25 and sum(s[1] == "OFF" for s in sessions) == 23,
            f"{df.person.nunique()} persons, {len(sessions)} sessions")
ck.internal("at most 20 passes per session (paper Methods: 20 trials)", max(s[2] for s in sessions) <= 20, f"max {max(s[2] for s in sessions)}, min {min(s[2] for s in sessions)}")
ck.internal("one row kept for every pass of every spatiotemporal workbook", len(df) == sum(s[2] for s in sessions), f"{len(df)} rows")
sp = df.walking_speed_mps
rng = (sp.between(0.2, 2) & df.stance_time_left_s.between(0.3, 2) & df.stance_time_right_s.between(0.3, 2)
       & df.step_length_left_m.between(0.05, 1.0) & df.step_length_right_m.between(0.05, 1.0))
nanrow = df[list(SPEC.values())].isna().any(axis=1)
ck.internal("at least 90% of passes have speed 0.2-2 m/s, stance 0.3-2 s and step lengths 0.05-1 m (the rest are kept as shared)",
            float(rng.mean()) > 0.9, f"{100 * rng.mean():.2f}% inside; speed {sp.min():.2f}-{sp.max():.2f} m/s; {int((sp < 0.2).sum())} passes under 0.2 m/s; "
            f"{int(nanrow.sum())} passes with an empty value; {int((df[list(SPEC.values())] == 0).any(axis=1).sum())} with a zero")
tot = (df.double_support_right_initial_s + df.double_support_right_terminal_s - df.double_support_s).abs()
ck.internal("double_support_s = initial + terminal double support of the right leg in the same pass", float(tot.median()) < 1e-6, f"median |diff| {tot.median():.2e}")

f = lambda k: pd.to_numeric(C(k), errors="coerce")  # noqa: E731
ck.published("participants", len(ids), 26, 0, "Participants")
ck.published("women", (C("gender") == "F").sum(), 6, 0, "Participants")
ck.published("age, minimum (44-81 years)", f("age").min(), 44, 0, "Participants")
ck.published("age, maximum", f("age").max(), 81, 0, "Participants")
ck.published("body mass, minimum (53.3-94.6 kg)", f("weight_kg").min(), 53.3, 0.05, "Participants")
ck.published("body mass, maximum", f("weight_kg").max(), 94.6, 0.05, "Participants")
ck.published("height, minimum (151.5-179.0 cm)", f("height_cm").min(), 151.5, 0.05, "Participants")
ck.published("height, maximum", f("height_cm").max(), 179.0, 0.05, "Participants")
ck.published("BMI, minimum (19.2-34.3)", f("bmi_kg_m2").min(), 19.2, 0.05, "Participants")
ck.published("BMI, maximum", f("bmi_kg_m2").max(), 34.3, 0.05, "Participants")
for k, (m1, s1, m0, s0) in {"moca": (22.80, 4.25, 23.35, 4.06), "updrs_iii": (24.80, 13.60, 26.43, 11.76), "fes_i": (27.72, 10.29, 33.91, 12.59)}.items():
    ck.published(f"{k.upper()} ON, mean", f("on_" + k).mean(), m1, 0.01, "Results (clinical characteristics)")
    ck.published(f"{k.upper()} ON, SD", f("on_" + k).std(), s1, 0.01, "Results (clinical characteristics)")
    ck.published(f"{k.upper()} OFF, mean", f("off_" + k).mean(), m0, 0.01, "Results (clinical characteristics)")
    ck.published(f"{k.upper()} OFF, SD", f("off_" + k).std(), s0, 0.01, "Results (clinical characteristics)")

meta = dict(
    paper=dict(authors="Shida TKF, Costa TM, de Oliveira CEN, de Castro Treza R, Hondo SM, Los Angeles E, Bernardo C, dos Santos de Oliveira L, de Jesus Carvalho M, Coelho DB",
               year=2023, title="A public data set of walking full-body kinematics and kinetics in individuals with Parkinson's disease",
               venue="Front Neurosci 17:992585", doi="10.3389/fnins.2023.992585"),
    lab="Coelho (Federal University of ABC, Brazil)", data_url="https://doi.org/10.6084/m9.figshare.14896881",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Kept every row of the deposit's *_spatiotemporal workbooks (one per participant and medication state) with participants "
             "coded; the participants file holds the numeric and categorical items of PDGinfo.xlsx (free-text medical history "
             "items are left out). The time-normalized joint-angle, ground-reaction-force and centre-of-mass curves and the raw "
             "C3D trials are not included."),
    how=["One row is one overground walking pass of one participant in one medication state (OFF or ON levodopa): the authors' "
         "means over the gait cycles of that pass. Times are in seconds, lengths in metres, speed in m/s and step and stride "
         "rates per minute, as in the deposit; stride_time_s is the deposit's cycle time and double_support_s its double limb "
         "support time (the sum of the right leg's initial and terminal double support).",
         "Rows are in the deposit's order (pass 1 to at most 20). The deposit does not say which medication state was walked "
         "first (the paper says the order was randomized), so session is 1 for both and block and condition give the state.",
         "Participants walked barefoot at a self-selected speed along a 20 m walkway with force plates in the floor."],
    unit_note="walking pass",
    differences=["The paper reports no group values for the spatiotemporal measures, so the checks are the participant and clinical "
                 "numbers of its Methods and Results. The paper's text says 13 participants had freezing of gait while its Results say no freezing "
                 "episodes occurred in the trials; the deposit's FoG group column is kept as shared."],
)
stdlib.release(DS, "walking", df, meta, ck, participants=part_people, participant_columns=pcols)
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
print(len(pcols), [c[0] for c in pcols])
