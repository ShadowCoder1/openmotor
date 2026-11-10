"""Kim, Schweighofer, Petzinger & Finley (2025) PLoS ONE 20(7):e0326692: learning to step over virtual obstacles with a target
foot clearance on a treadmill, people with early Parkinson's disease (15) and age-matched controls (20). One row per obstacle
crossing, from the five tables of the OSF deposit (Base_table, LL_Table, Catch_table, Retention_table, Demographics_Share).

Raw files: raw_extra/Kim2025_PLoSOne/OSF.zip (OSF project d3m6z, no license stated). Read with pandas only.
Run from the site folder: python3 -I tools/std/convert/Kim2025_PLoSOne.py"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Kim2025_PLoSOne"
raw = stdlib.raw_dir(DS)
z = zipfile.ZipFile(raw / "OSF.zip")
X = lambda n, **k: pd.read_excel(io.BytesIO(z.read(f"OSF/{n}.xlsx")), **k)
B, C, R, L = X("Base_table"), X("Catch_table"), X("Retention_table"), X("LL_Table")
demo = X("Demographics_Share", sheet_name=None)
ck = stdlib.Checks()

# ---- participants: PD sheet ids equal the data ids; control sheet (HO) ids + 50 equal the data ids
pd_d, ho_d = demo["PD"].copy(), demo["HO"].copy()
pd_d["sid"], ho_d["sid"] = pd_d.Sub, ho_d.Sub + 50
pd_d["grp"], ho_d["grp"] = 1, 2
allsubs = sorted(set(B["sub"]))
dem = pd.concat([pd_d, ho_d]).set_index("sid")
dem = dem.loc[[s for s in allsubs if s in dem.index]]
assert set(dem.index) == set(allsubs) and len(dem) == 35
grp_of = dict(zip(B["sub"], B["group"]))
assert all(dem.grp[s] == grp_of[s] for s in dem.index), "group of demographics sheet differs from data tables"
GROUP = {1: "Parkinson's disease", 2: "control"}
POP = {1: "Parkinson's disease", 2: "healthy"}
SEX = {1: "M", 2: "F"}  # README of the deposit: Sex 1 male, 2 female
DEVICE = ("Treadmill at self-selected speed with a head-mounted display (virtual obstacles), harness and handrails; "
          "Qualisys motion capture at 100 Hz; auditory feedback on foot clearance")


def side(v):
    return np.where(pd.isna(v), "", np.where(v == 1, "left", "right"))


def mk(df, sub, bout, block, cond, feedback, session, onum, pos, fh, cl, col, oh, err, side_v=None, alt=None, grp=None):
    out = pd.DataFrame(dict(
        dataset=DS, study=DS, experiment="", sub=df[sub].values, group=df[grp].map(GROUP).values if grp else "",
        block=block, phase=bout, condition=cond, feedback=feedback, session=session,
        obstacle_number=df[onum].values, obstacle_position_m=df[pos].values, foot_height_m=df[fh].values if fh else np.nan,
        foot_clearance_m=df[cl].values, obstacle_collision=df[col].values, obstacle_height_m=df[oh].values))
    out["performance_error_m"] = df[err].values if err else np.nan
    out["performance_error_alt_m"] = df[alt].values if alt else np.nan
    out["obstacle_side"] = side(df[side_v]) if side_v else ""
    return out


parts = []
b = mk(B, "sub", "baseline", "baseline", "baseline (collision sounds only)", "none", 1, "obnum", "obx", "footheight", "clearance", "col", "obheight", "error", "side", grp="group")
b["seq"] = 0
parts.append(b)
for t in sorted(L.trial.unique()):
    d = L[L.trial == t]
    x = mk(d, "subject", "learning", f"practice bout {t}", "skill acquisition with auditory feedback", "altered", 1, "obnumb", "obpos", "clear", "clear_height", "collision", "obheight", None, grp="groupindex")
    x["seq"] = t
    parts.append(x)
# catch rows: the deposit labels the block 1 (most participants) or 7 (two); the protocol puts the no-feedback bout 6th
c = mk(C, "sub", "test", C.block.map(lambda v: f"catch (deposit block {v})"), "no performance feedback (catch bout)", "none", 1, "obnum", "obx", "footheight", "clearance", "col", "obheight", "error", "side", grp="group")
c["seq"] = 6
parts.append(c)
r = mk(R, "sub", "test", R.block.map(lambda v: f"retention block {v}"), "overnight retention, no performance feedback", "none", 2, "obnum", "obx", "footheight", "clearance", "col", "obheight", "error", "side", alt="error.1", grp="group")
r["seq"] = 100
parts.append(r)
df = pd.concat(parts, ignore_index=True)
df["orig"] = np.arange(len(df))
df = df.sort_values(["sub", "seq", "orig"], kind="stable").reset_index(drop=True)
df["trial"] = df.groupby("sub").cumcount() + 1
df["subject"] = "S" + df["sub"].astype(str)
df["obstacle_collision"] = df.obstacle_collision.map(lambda v: "" if pd.isna(v) else bool(v))
for k, v in dict(subject_excluded_by_authors=False, population="", setting="lab", device=DEVICE, task="walking", unit="step",
                 pert_type="none", pert_value=0.0, pert_unit="", excluded_by_authors=False, handedness="", time_s=np.nan,
                 walking_context="treadmill", surface_incline_deg=0.0).items():
    df[k] = v
df["population"] = df["sub"].map(lambda s: POP[grp_of[s]])
df["group"] = df["sub"].map(lambda s: GROUP[grp_of[s]])
df["age"] = df["sub"].map(dem.Age)
df["sex"] = df["sub"].map(dem.Sex_m_1).map(SEX)
df["belt_speed_fast_mps"] = df["sub"].map(dem.Speed)
df["belt_speed_slow_mps"] = df["belt_speed_fast_mps"]
df["fast_leg"] = "unknown"
df["subject"] = "S" + df["sub"].astype(str)
sub_col = df["sub"].values
df = df.reindex(columns=[c[1] for c in stdlib.columns_for("walking") if c[1] != "person"])

# ---- participants.csv
pcols = [("group", "text", "", "Parkinson's disease or control."),
         ("treadmill_speed_mps", "number", "m/s", "Self-selected treadmill walking speed used in the task (Speed in the deposit)."),
         ("overground_speed_mps", "number", "m/s", "Overground walking speed (OvergroundS in the deposit; for controls the deposit's 10MWT_v column holds the same values)."),
         ("mini_bestest", "number", "points", "Mini-BESTest score (MiniBest); empty when missing in the deposit."),
         ("mds_updrs_iii", "number", "points", "Movement Disorder Society UPDRS part III (UPDRS in the deposit); Parkinson's group only."),
         ("hoehn_yahr", "integer", "", "Hoehn and Yahr stage (HY); Parkinson's group only."),
         ("years_since_diagnosis", "number", "years", "Years since diagnosis (Yrs_diagnosis); Parkinson's group only."),
         ("levodopa_equivalent_mg", "number", "mg/day", "Levodopa equivalent daily dose (LEDD); Parkinson's group only.")]
part = []
for sid, r_ in dem.iterrows():
    part.append(dict(person="S" + str(sid), group=GROUP[r_.grp], treadmill_speed_mps=r_.Speed, overground_speed_mps=r_.OvergroundS,
                     mini_bestest=r_.MiniBest, mds_updrs_iii=r_.get("UPDRS"), hoehn_yahr=r_.get("HY"),
                     years_since_diagnosis=r_.get("Yrs_diagnosis"), levodopa_equivalent_mg=r_.get("LEDD")))

# ---- checks
ck.internal("all rows of the four tables kept", len(df) == len(B) + len(C) + len(R) + len(L) == 1102 + 1072 + 2204 + 5600)
ck.internal("35 participants in the data tables and the demographics", df.subject.nunique() == 35 == len(dem))
ck.internal("baseline bout has at most 32 obstacles per participant", df[df.phase == "baseline"].groupby("subject").size().max() <= 32)
ck.internal("obstacle heights are 0.05 or 0.18 m", set(df.obstacle_height_m.dropna().round(3)) == {0.05, 0.18})
gap = df.foot_height_m - df.obstacle_height_m
isl = df.phase == "learning"
ok_signed = ((gap - df.foot_clearance_m).abs()[~isl].dropna() < 1e-6).all()
ok_abs = ((gap.abs() - df.foot_clearance_m).abs()[isl].dropna() < 1e-6).all()
ck.internal("clearance = foot height minus obstacle height (signed) in the baseline, catch and retention tables", bool(ok_signed))
ck.internal("clearance = absolute value of that difference in the practice-bout table", bool(ok_abs))
N_NEG = int((gap[isl] < 0).sum())
ck.published("participants analysed (37 recruited, 2 controls excluded)", df.subject.nunique(), 35, 0, "Methods (Participants)")
ck.published("people with Parkinson's disease", len(dem[dem.grp == 1]), 15, 0, "Methods (Participants)")
ck.published("obstacle heights (low, m)", df.obstacle_height_m.min(), 0.05, 0.005, "Methods (Setup)")
ck.published("obstacle heights (high, m)", df.obstacle_height_m.max(), 0.18, 0.005, "Methods (Setup)")
P = dem[dem.grp == 1]
ck.published("PD age, mean", P.Age.mean(), 65, 0.5, "Table 1")
ck.published("PD age, SD", P.Age.std(), 10, 0.5, "Table 1")
ck.published("PD men", int((P.Sex_m_1 == 1).sum()), 7, 0, "Table 1")
ck.published("PD women", int((P.Sex_m_1 == 2).sum()), 8, 0, "Table 1")
ck.published("PD treadmill speed, mean", P.Speed.mean(), 1.00, 0.005, "Table 1")
ck.published("PD treadmill speed, SD", P.Speed.std(), 0.24, 0.005, "Table 1")
ck.published("PD overground speed, mean", P.OvergroundS.mean(), 1.35, 0.005, "Table 1")
ck.published("PD Mini-BESTest, mean", P.MiniBest.mean(), 24, 0.5, "Table 1")
ck.published("PD MDS-UPDRS III, mean", P.UPDRS.mean(), 20, 0.5, "Table 1")
ck.published("PD MDS-UPDRS III, SD", P.UPDRS.std(), 8, 0.5, "Table 1")
ck.published("PD LED, mean (mg)", P.LEDD.mean(), 520, 0.5, "Table 1")
ck.published("PD LED, SD (mg)", P.LEDD.std(), 377, 0.5, "Table 1")
ck.published("PD Hoehn and Yahr stage 1", int((P.HY == 1).sum()), 2, 0, "Table 1")
ck.published("PD Hoehn and Yahr stage 2", int((P.HY == 2).sum()), 13, 0, "Table 1")

meta = dict(
    paper=dict(authors="Kim A, Schweighofer N, Petzinger GM, Finley JM", year=2025,
               title="Uncovering locomotor learning dynamics in people with Parkinson's disease",
               venue="PLoS ONE 20(7):e0326692", doi="10.1371/journal.pone.0326692"),
    lab="Finley (University of Southern California)", data_url="https://osf.io/d3m6z/",
    license="none stated", license_url="",
    changes=("Kept every row of Base_table, LL_Table, Catch_table and Retention_table, regrouped by participant in the order of the "
             "session (baseline, practice bouts 2-5, catch bout, practice bout 7, next-day retention); participants coded; "
             "age and sex from Demographics_Share joined on; units are the authors' (metres)."),
    how=["One row is one crossing of a virtual obstacle while walking on a treadmill: the obstacle (height 0.05 or 0.18 m, "
         "position along the corridor in metres, and for most tables its side), the toe-marker height, the clearance over the obstacle "
         "(foot height minus obstacle height), whether the foot collided, and the performance error (distance of the clearance from the "
         "nearest edge of the hidden success range for that height). Bouts have 30 or 32 obstacles.",
         "Table to protocol: Base_table is the baseline; LL_Table holds practice bouts numbered 2, 3, 4, 5 and 7 with performance feedback "
         "(no side or error columns); Catch_table is the no-feedback bout between bouts 5 and 7 (the paper's 'second to last bout'); "
         "Retention_table is the next-day test without performance feedback (collision sounds only). This mapping, and 'LL' meaning the "
         "learning bouts, is inferred from the paper's Methods and the bout numbers; the deposit's readme does not explain LL or the bouts. "
         "In the deposit the catch bout of two participants is labelled block 7 instead of 1.",
         "Sex is coded 1 male, 2 female by the readme; the 7 men and 8 women of the Parkinson's group match the paper's Table 1. "
         "Control ids in the demographics file are the data ids minus 50. feedback is 'altered' for bouts with auditory performance "
         "feedback and 'none' where only collision sounds were played. belt_speed columns hold the participant's self-selected treadmill "
         "speed (both belts equal)."],
    unit_note="obstacle crossing",
    differences=[f"In the practice-bout table (LL_Table) the authors' clearance is the absolute value of foot height minus obstacle height ({N_NEG} rows where the foot was lower than the top of the obstacle show a positive number), whereas in the baseline, catch and retention tables it is signed (negative when lower). foot_clearance_m is the authors' value as shared in each table, so signs are not comparable across them.",
                 "Retention_table of 18 control participants lists the first 32 obstacles of the retention test twice (identical obstacle "
                 "positions and foot heights; the repeat has error 1.0 in the authors' error column, while their undocumented second error "
                 "column error.1 holds the real value). Both listings are kept; performance_error_alt_m holds error.1. Treat the repeats "
                 "(the second 32 rows with block 1 and trial numbers 1-32 again) as duplicates in analysis.",
                 "Table 1 of the paper gives the control group for 22 participants; the deposit's 20 analysed controls are not "
                 "comparable to it, so control demographics are not used as checks."])
stdlib.release(DS, "walking", df.assign(subject=df.subject), meta, ck, participants=part, participant_columns=pcols)
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
