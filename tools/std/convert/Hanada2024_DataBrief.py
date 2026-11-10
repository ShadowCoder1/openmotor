"""Hanada, Kalabic & Ferris (2024) Data in Brief: mobile brain/body imaging data set of indoor treadmill walking and outdoor
walking with a visual search task (UM-LSIE, Figshare 10.6084/m9.figshare.6741734, CC BY 4.0). Only the authors' gait events item
(Figshare 10.6084/m9.figshare.27022414, LSIE_gait_events.zip, CC BY 4.0) is converted: one row per stride.

The per-subject raw folders (EEG, IMU, eye tracking, force plates, video; 836 GB) are far over the size limit and are not converted.
Raw files (raw_extra/Hanada2024_DataBrief/): LSIE_gait_events.zip (80 MAT-files v5, one per participant and environment, a struct array
'events' with type, latency in EEG samples at 512 Hz, hedtags, urevent), Cortisol_all_subjects.xlsx (per-sample table that also holds sex,
age, height, weight and the order of environments and conditions of each participant) and UM_LSIE.xml (read as text).
Run from the site folder: python3 -I tools/std/convert/Hanada2024_DataBrief.py"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Hanada2024_DataBrief"
raw = stdlib.raw_dir(DS)
FS = 512.0
ck = stdlib.Checks()

x = pd.read_excel(raw / "Cortisol_all_subjects.xlsx")
x.columns = ["subid", "sex", "age", "height_in", "weight_lb", "env", "order_env", "order_stress", "cond", "conc", "cond_ordered"]
demo = x.drop_duplicates("subid").set_index("subid")

z = zipfile.ZipFile(raw / "LSIE_gait_events.zip")
names = sorted(n for n in z.namelist() if n.endswith(".mat"))
PAT = re.compile(r"events/(LSIE_\d+)_(Indoor|Outdoor)(?:_part(\d+))?_HED\+Gait\.mat")
parsed = {n: PAT.fullmatch(n).groups() for n in names}  # a session split over several EEG files has _partN files
assert all(PAT.fullmatch(n) for n in names)
subs = sorted({g[0] for g in parsed.values()})
CONDS = ["Baseline", "Non-stress", "Stress"]
DEVICE = {"Indoor": "Bertec split-belt instrumented treadmill in front of a virtual-reality park screen (belts at the same speed); 256-channel EEG, "
                    "6 foot/ankle/waist/chest IMUs; gait events from the treadmill force plates (vertical force)",
          "Outdoor": "Arboretum trail walk with a 256-channel EEG in a backpack; gait events from the inertial measurement units on each foot"}


def markers(ev):
    """(latency in samples, condition, is_onset) for the Baseline / Non-stress / Stress onset and offset markers."""
    out = []
    for e in ev:
        t = str(e.type)
        if t in CONDS:
            tags = str(e.hedtags)
            out.append((int(e.latency), t, "Attribute/Onset" in tags))
    return out


rows, st = [], dict(outside=0, no_to=0, files=0)
seg_len = {c: [] for c in CONDS}
order_cache = {}
for sid in subs:
    n_trial = 0
    d = demo.loc[sid] if sid in demo.index else None
    n_trial = 0
    envs = sorted({parsed[n][1] for n in names if parsed[n][0] == sid})
    if d is not None:
        first = "Outdoor" if int(d.order_env) == 1 else "Indoor"
        envs = sorted(envs, key=lambda e: 0 if e == first else 1)
    for si, env in enumerate(envs, start=1):
        parts_of = sorted((int(parsed[n][2] or 1), n) for n in names if parsed[n][0] == sid and parsed[n][1] == env)
        active = None  # condition in force at the end of the previous part
        for part, n in parts_of:
            m = sio.loadmat(io.BytesIO(z.read(n)), squeeze_me=True, struct_as_record=False)["events"]
            st["files"] += 1
            mk = sorted(markers(m), key=lambda q: (q[0], q[2]))  # offsets before onsets at the same sample
            starts = {c: [q[0] for q in mk if q[1] == c and q[2]] for c in CONDS}
            ends = sorted(q[0] for q in mk if not q[2])
            for c in CONDS:
                seg_len[c] += [(min([x for x in ends if x > a0] or [a0]) - a0) / FS / 60 for a0 in starts[c]]

            def state(t, active=active, mk=mk):
                cur = active
                for lat, c, on in mk:
                    if lat > t:
                        break
                    cur = c if on else None
                return cur

            ty = np.array([str(e.type) for e in m])
            lat = np.array([float(e.latency) for e in m]) / FS
            recs = []
            for foot, side in (("R", "right"), ("L", "left")):
                hs = np.sort(lat[ty == f"{foot}HS"])
                to = np.sort(lat[ty == f"{foot}TO"])
                for a, b in zip(hs[:-1], hs[1:]):
                    cond = state(a * FS)
                    nxt = [x for x in ends if x > a * FS]
                    if cond is None or (nxt and nxt[0] <= b * FS):
                        st["outside"] += 1
                        continue
                    inside = to[(to > a) & (to < b)]
                    r = dict(side=side, start=a, stride=b - a, cond=cond)
                    if len(inside) == 1:
                        r.update(stance=inside[0] - a, swing=b - inside[0])
                    else:
                        st["no_to"] += 1
                    recs.append(r)
            active = state(float("inf"))
            for r in sorted(recs, key=lambda q: q["start"]):
                s_ = r["side"]
                n_trial += 1
                stress = r["cond"] == "Stress"
                dd = dict(
                    dataset=DS, study=DS, experiment=f"{env.lower()} " + ("treadmill walking" if env == "Indoor" else "walking on an arboretum trail"),
                    group="", subject=sid, subject_excluded_by_authors=False,
                    age=float(d.age) if d is not None else np.nan, sex=({1: "M", 2: "F"}[int(d.sex)] if d is not None else ""),
                    handedness="", population="healthy", setting="lab" if env == "Indoor" else "home", device=DEVICE[env], session=si,
                    block=r["cond"], trial=n_trial, time_s=r["start"], phase="baseline" if r["cond"] == "Baseline" else "test", task="walking",
                    unit="stride", condition=f"{r['cond']} (visual search task)" if r["cond"] != "Baseline" else "Baseline (no flags)",
                    pert_type="induced_stress", pert_value=float(stress),
                    pert_unit="0 no, 1 yes (money penalties and sirens during the visual search)", feedback="normal", excluded_by_authors=False,
                    fast_leg="unknown", walking_trial=part, stride_start_s=r["start"],
                    walking_context="treadmill" if env == "Indoor" else "overground")
                if env == "Indoor":
                    dd.update(belt_speed_fast_mps=0.7, belt_speed_slow_mps=0.7, surface_incline_deg=0.0)
                dd[f"stride_time_{s_}_s"] = r["stride"]
                if "stance" in r:
                    dd[f"stance_time_{s_}_s"] = r["stance"]
                    dd[f"swing_time_{s_}_s"] = r["swing"]
                    dd[f"stance_pct_{s_}"] = 100 * r["stance"] / r["stride"]
                    dd[f"swing_pct_{s_}"] = 100 * r["swing"] / r["stride"]
                rows.append(dd)
df = pd.DataFrame(rows).reindex(columns=[c[1] for c in stdlib.columns_for("walking") if c[1] != "person"])
for c in ("time_s", "stride_start_s", "stride_time_left_s", "stride_time_right_s", "stance_time_left_s", "stance_time_right_s",
          "swing_time_left_s", "swing_time_right_s"):
    df[c] = df[c].round(6)

have = demo.loc[[s for s in subs if s in demo.index]]
participants = pd.DataFrame(dict(
    person=have.index, height_m=(have.height_in * 0.0254).round(4).values, weight_kg=(have.weight_lb * 0.45359237).round(2).values,
    environment_first=["outdoor" if v == 1 else "indoor" for v in have.order_env],
    stress_condition_first=[bool(v == 1) for v in have.order_stress]))
pcols = [("height_m", "number", "m", "Height, converted from the inches in the deposit's cortisol spreadsheet (x 0.0254)."),
         ("weight_kg", "number", "kg", "Body mass, converted from the pounds in the deposit's cortisol spreadsheet (x 0.45359237)."),
         ("environment_first", "text", "indoor, outdoor", "Environment walked first (the spreadsheet's orderenvironment)."),
         ("stress_condition_first", "true/false", "", "The stress condition came before the non-stress condition in both environments (the spreadsheet's orderstress).")]

nsub = df.subject.nunique()
ck.internal("every gait-event file in the ZIP read", st["files"] == len(names) and nsub > 0, str(st))
sh = float(((df.stride_time_left_s.fillna(df.stride_time_right_s) > 0.4) & (df.stride_time_left_s.fillna(df.stride_time_right_s) < 3)).mean())
ck.internal("stride times between 0.4 and 3 s for 98% of strides", sh > 0.98, f"{sh:.4f}")
ck.internal("every participant has strides of both feet in every session",
            all(g.groupby("session").apply(lambda q: q.stride_time_left_s.notna().any() and q.stride_time_right_s.notna().any()).all() for _, g in df.groupby("subject")))
ck.published("youngest participant, years (ages 18-45)", x.drop_duplicates("subid").age.min(), 18, 0, "Section 4.1")
ck.published("oldest participant, years (ages 18-45)", x.drop_duplicates("subid").age.max(), 45, 0, "Section 4.1")
for c in CONDS:
    if seg_len[c]:
        print(c, "segment minutes: n", len(seg_len[c]), "mean", np.mean(seg_len[c]), "median", np.median(seg_len[c]))
print("subjects", len(subs), "files in zip", len(names), "sheet-subs without events", sorted(set(demo.index)-set(subs)), "events-subs not in sheet", sorted(set(subs)-set(demo.index)))
print(st, len(df), df.groupby("session").size().to_dict(), "sex", have.sex.value_counts().to_dict(), "mean age", have.age.mean(), "all-47 mean age", demo.age.mean())

meta = dict(
    paper=dict(authors="Hanada GM, Kalabic M, Ferris DP", year=2024,
               title="Mobile brain-body imaging data set of indoor treadmill walking and outdoor walking with a visual search task",
               venue="Data Brief 57:110968", doi="10.1016/j.dib.2024.110968"),
    lab="Ferris (University of Michigan / University of Florida)", data_url="https://doi.org/10.6084/m9.figshare.27022414",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("From the authors' gait events (heel strikes and toe-offs; latencies in EEG samples at 512 Hz, divided by 512 to give seconds) "
             "the stride, stance and swing times of each stride were computed as differences of event times. A stride is kept only when "
             "both of its heel strikes fall inside one of the three labelled conditions (Baseline, Non-stress, Stress). The EEG, IMU, "
             "force-plate, eye-tracking, heart-rate and video recordings and the cortisol values are not included."),
    how=["One row is one stride of one foot (heel strike to the next heel strike of the same foot) of one participant in one environment. "
         "Left strides fill the _left columns and right strides the _right columns; rows are in time order within each environment "
         "(session 1 is the environment the participant walked first, from the spreadsheet's order column). time_s and stride_start_s "
         "are the time of the starting heel strike from the start of the EEG recording of that session. Stance is the foot's heel strike "
         "to its toe-off and swing the toe-off to the next heel strike (seconds); the percentages are of the stride time.",
         "Each environment had three walking conditions of about 20 minutes: a baseline walk without flags, then a non-stress and a stress "
         "visual search condition in random order. The stress condition added a USD 1 penalty and a siren for each missed target flag and "
         "random false sirens, so pert_type is induced_stress (pert_value 1 in the stress condition). The condition of a stride comes from "
         "the onset and offset markers of the same name in the authors' event list. Indoors the treadmill ran at 0.7 m/s (paper, section "
         "4.10.1; the event list labels the treadmill marker 0.8, which is not used); outdoors the pace was self-selected and the belt "
         "speed columns are empty.",
         "Indoor events come from the treadmill force plates and outdoor events from the foot IMUs, as the paper states. Left and right "
         "strides can differ in number because the authors' event detection missed or added events, and the deposit has no further "
         "quality information. Sex and age come from the deposit's cortisol spreadsheet; the sex codes 1 = male and 2 = female are as "
         "the paper lists them."],
    unit_note="stride (of one foot)",
    differences=["The gait-events item covers 39 of the 49 participants (80 events files; 7 participants have one environment only and some sessions are split in parts) and the "
                 "cortisol spreadsheet, the only per-participant demographics, lists 47 (one participant with events, LSIE_14, is not in it, so that participant has no age or sex). The paper's 49 participants (20 men, 29 women, mean age "
                 "22.7) therefore cannot be counted from the deposit: the spreadsheet gives 19 men, 28 women and mean age 22.8 for 47, so those "
                 "numbers are not used as checks; its age range 18-45 is reproduced.",
                 "Strides whose two heel strikes are not both inside a labelled condition (before the baseline marker, between conditions, "
                 "or in a file missing a condition marker) are left out.",
                 "The paper's and the deposit's other outcomes (EEG, cortisol, visual search performance) are not part of the converted file."],
)
stdlib.release(DS, "walking", df, meta, ck, participants=participants, participant_columns=pcols)
for c in ck.items:
    print(c.get("what"), c["ok"], c.get("computed"), c.get("reported"))
