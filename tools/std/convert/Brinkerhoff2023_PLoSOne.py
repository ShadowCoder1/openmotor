"""Brinkerhoff, Sanchez & Roper (2023) PLoS ONE 18(6):e0286649: 10 min of split-belt walking (2:1) in 32 young adults who
exercised at least 150 min/week (MOVE, 19) or less (notMOVE, 13).

Raw files: raw_extra/Brinkerhoff2023_PLoSOne/ sla.csv (Figshare 10.6084/m9.figshare.22654540, "exercise&adapt_sla.csv") and
work.csv (Figshare 10.6084/m9.figshare.22659499, "exercise&adapt_work.csv"), both CC BY 4.0, listed in the paper's Figshare
collection 10.6084/m9.figshare.c.6607117. The authors' R script (item 22661266) was read as text only.
Run from the site folder: python3 -I tools/std/convert/Brinkerhoff2023_PLoSOne.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Brinkerhoff2023_PLoSOne"
raw = stdlib.raw_dir(DS)
GROUP = {"move": "MOVE (at least 150 min/week of exercise)", "notmove": "notMOVE (less than 150 min/week of exercise)"}
DEVICE = ("Instrumented split-belt treadmill, 100 Hz motion capture and 1000 Hz ground reaction forces; fast belt at the "
          "participant's fastest comfortable walking speed, slow belt at half of it")
sla = pd.read_csv(raw / "sla.csv")
wk = pd.read_csv(raw / "work.csv")
ck = stdlib.Checks()
order = sla.id.drop_duplicates().tolist()
assert set(order) == set(wk.id) and wk.id.drop_duplicates().tolist() == order
info = wk.drop_duplicates("id").set_index("id")
FL = {1: "right", 2: "left"}


def base(d, unit, cond):
    i = info.loc[d.id]
    return dict(dataset=DS, study=DS, experiment="", group=d.group.map(GROUP).values, subject=d.id.values,
                subject_excluded_by_authors=False, population="healthy", setting="lab", device=DEVICE, session=1,
                block="Adapt", phase="learning", task="walking", unit=unit, condition=cond,
                pert_type="split_belt", pert_value=2.0, pert_unit="ratio (fast:slow belt speed)", feedback="normal",
                excluded_by_authors=False, fast_leg=i.fast_leg.map(FL).values, belt_speed_fast_mps=i.fast_speed.values,
                belt_speed_slow_mps=(i.fast_speed / 2).values, walking_context="treadmill", surface_incline_deg=0.0)


a = pd.DataFrame(base(sla, "step", "step length asymmetry series"))
a["step_length_asymmetry"] = sla.sla.values
b = pd.DataFrame(base(wk, "stride", "work rate series"))
for col, src in [("work_rate_positive_fast_wkg", "wfposrate_norm"), ("work_rate_negative_fast_wkg", "wfnegrate_norm"),
                 ("work_rate_positive_slow_wkg", "wsposrate_norm"), ("work_rate_negative_slow_wkg", "wsnegrate_norm"),
                 ("work_rate_net_wkg", "wnetrate_norm")]:
    b[col] = wk[src].values
df = pd.concat([a, b], ignore_index=True)
df["subject"] = df.subject.astype(str)
df["sub_order"] = df.subject.map({s: i for i, s in enumerate(order)})
df = df.sort_values("sub_order", kind="stable").drop(columns="sub_order").reset_index(drop=True)
df["trial"] = df.groupby("subject").cumcount() + 1
df["time_s"] = None
df = df.reindex(columns=[c[1] for c in stdlib.columns_for("walking") if c[1] != "person"])

part, pcols = [], [
    ("leg_length_m", "number", "m", "Leg length (LL_m in the work file)."),
    ("rpe_end_of_adaptation", "number", "6-20 scale", "Rating of perceived exertion (Borg 6-20) taken at the end of the adaptation trial; empty for 2 participants."),
    ("net_work_rate_last100_wkg", "number", "W/kg", "Net mechanical work rate averaged over the last 100 strides (avg100workratenet_norm in the deposit).")]
for s in order:
    r = info.loc[s]
    rp = sla[sla.id == s].RPE.dropna()
    part.append(dict(person=s, leg_length_m=r.LL_m, rpe_end_of_adaptation=rp.iloc[0] if len(rp) else None,
                     net_work_rate_last100_wkg=r.avg100workratenet_norm))

g = sla.drop_duplicates("id").group.value_counts()
ck.internal("rows kept: 26,336 step rows (32 x 823) and 13,216 stride rows (32 x 413), none dropped",
            len(a) == 26336 and len(b) == 13216 and len(df) == 39552)
ck.internal("RPE is one value per participant", (sla.groupby("id").RPE.nunique() <= 1).all())
ck.published("participants in total", len(order), 32, 0, "Methods (32 analysed after 5 exclusions)")
ck.published("MOVE participants", g["move"], 19, 0, "Methods (Participants)")
ck.published("notMOVE participants", g["notmove"], 13, 0, "Methods (Participants)")
ck.published("steps per participant", sla.groupby("id").size().iloc[0], 823, 0, "Methods (data truncated to 823 steps)")
net = {k: info[info.group == k].avg100workratenet_norm for k in GROUP}
# the paper's MOVE / notMOVE net work rates match the deposit's groups the other way round (see differences)
for k, rk, rm, rs in [("notmove", "MOVE", 0.166, 0.072), ("move", "notMOVE", 0.188, 0.085)]:
    ck.published(f"{rk} net work rate, last 100 strides, mean (deposit group '{k}')", net[k].mean(), rm, 0.0015, "Results (W/kg)")
    ck.published(f"{rk} net work rate, last 100 strides, SD (deposit group '{k}')", net[k].std(), rs, 0.0015, "Results (W/kg)")
first = sla[sla.step <= 5]
for k, rep in [("move", -0.144), ("notmove", -0.137)]:
    ck.published(f"{k} initial SLA (first 5 steps), mean", first[first.group == k].groupby("id").sla.mean().mean(), rep, 0.01, "Results (initial SLA)")
meta = dict(
    paper=dict(authors="Brinkerhoff SA, Sanchez N, Roper JA", year=2023,
               title="Habitual exercise evokes fast and persistent adaptation during split-belt walking",
               venue="PLoS ONE 18(6):e0286649", doi="10.1371/journal.pone.0286649"),
    lab="Roper (Auburn University)", data_url="https://doi.org/10.6084/m9.figshare.c.6607117.v2",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Kept every row of both Figshare files in their order, with participants coded and groups named in the authors' words "
             "(the deposit's 'move' and 'notmove'). The step and stride series share a participant and follow each other (steps first). "
             "The group- and mean-centered SLA columns (gc_sla, c_sla) are derived values and are not included; RPE, leg length and "
             "the last-100-stride mean work rate are in participants.csv."),
    how=["The table holds two series for each participant, told apart by unit and condition. Rows with unit step (823 per person) are the "
         "authors' normalized step length asymmetry for each step of the 10-minute split-belt adaptation trial (step_length_asymmetry, "
         "unitless; negative when the slow-belt leg takes the longer step). Rows with unit stride (413 per person) are the authors' "
         "mechanical work rates of the fast and slow leg (positive, negative) and the net, in W/kg after the authors' normalization.",
         "Only the adaptation trial was shared (the 2:1 split-belt trial, fast belt at the participant's fastest comfortable speed in m/s, "
         "slow belt half of it); no baseline or washout rows, no age or sex per participant. fast_leg is from the work file (1 = right, 2 = left) "
         "and applied to the step rows of the same participant.",
         "The paper's SLA equation is an image; the deposit calls the column step length asymmetry normalized and the paper normalizes it by "
         "stride length, which is the module's (fast - slow) / (fast + slow) quantity for consecutive steps."],
    unit_note="step (unit step) or stride (unit stride)",
    differences=[
        "The paper reports last-100-stride net work rates of 0.166 (SD 0.072) W/kg for MOVE and 0.188 (SD 0.085) for notMOVE. In the "
        "deposit the 19 'move' participants have 0.188 (SD 0.085) and the 13 'notmove' participants 0.166 (SD 0.072): the numbers match "
        "exactly but with the group labels the other way round. The group sizes (19, 13) and the initial step length asymmetry "
        "(-0.153 and -0.137 here, -0.144 and -0.137 in the paper) agree with the labels as given, so the labels are kept as in the "
        "deposit and the swap is left for the authors to explain (the paper text or the work file may be mislabelled).",
        "The initial work rates of the paper (first five strides, for example slow-leg negative work -1.664 and -1.57 W/kg) come out "
        "about 0.05 W/kg more negative here (-1.727 and -1.619), and the deposit's avg100workratenet_norm differs from the mean of the "
        "last 100 net work rates in the stride rows for some participants; the authors' own value is kept in participants.csv."])
stdlib.release(DS, "walking", df, meta, ck, participants=part, participant_columns=pcols)
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
