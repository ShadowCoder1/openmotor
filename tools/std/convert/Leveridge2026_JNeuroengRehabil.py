"""Leveridge, Russo, Williams, Ye, Wang, Lamb & Young (2026) J Neuroeng Rehabil 23(1):221: head, torso and pelvis rotation in
360-degree turns on the spot, people with Parkinson's disease with and without freezing of gait and healthy older adults (46 analysed).
One row per turn (recording): the unwrapped transverse-plane angles of head, torso and pelvis (100 Hz, cropped as the authors describe),
with the turn time, turn angle and heel-strike events they computed; participants.csv holds the SPSS table (demographics, clinical
scores and the authors' per-participant outcomes).
Raw: raw_extra/Leveridge2026_JNeuroengRehabil/ anglesexpanded_UNWRAP.mat, turntime.mat, step_count_curvature.mat, SPSS_main.sav (OSF ydb4z, CC BY 4.0).
Not carried: MOCAP_Insoles_Data.mat (marker table object and insole pressure columns that the deposit does not describe).
Run from the site folder: python3 -I tools/std/convert/Leveridge2026_JNeuroengRehabil.py"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Leveridge2026_JNeuroengRehabil"
D = stdlib.raw_dir(ID)
TMP = Path("tmp_" + ID)
shutil.rmtree(TMP, ignore_errors=True)
(TMP / "samples").mkdir(parents=True)
ld = lambda f, k: sio.loadmat(D / f, squeeze_me=True, struct_as_record=False)[k]
H, P, T = (ld("anglesexpanded_UNWRAP.mat", k) for k in ("head_angles_UNWRAP", "pelvis_angles_UNWRAP", "torso_angles_UNWRAP"))
TT = ld("turntime.mat", "turntimestructure")
SC = ld("step_count_curvature.mat", "stepcount_curvature")
sp, _ = pyreadstat.read_sav(D / "SPSS_main.sav")
GROUP = {1: "PD with freezing of gait", 2: "PD without freezing of gait", 3: "healthy older adults"}
POP = {1: "Parkinson's disease with freezing of gait", 2: "Parkinson's disease without freezing of gait", 3: "healthy"}
HZ = 100.0

# the SPSS table is linked to the MATLAB participant keys through the mean turn time (unique for all 46 participants)
mean_tt = {k: float(getattr(TT, k).mean) for k in TT._fieldnames}
key_of = {}
for i, r in sp.iterrows():
    c = [k for k in TT._fieldnames if abs(mean_tt[k] - r.turn_time) < 0.0015]
    assert len(c) == 1, (r.pptID, c)
    key_of[i] = c[0]
assert len(set(key_of.values())) == 46 and set(key_of.values()) == set(H._fieldnames)
sp["key"] = [key_of[i] for i in sp.index]
sp["code"] = ["G%d_%02d" % (int(g), n + 1) for n, g in enumerate(sp.GroupID)]  # replaced by stdlib; keeps the group in the working label
key2row = {r.key: r for _, r in sp.iterrows()}
order = sp.sort_values(["GroupID", "pptID"]).key.tolist()

rows, extra = [], []
stat = dict(turns=0, short=0, nan_inside=0, events_out=0, hs_count_ok=True, n_per_dir={})
DEVICE = "19-camera OptiTrack Prime 13 motion capture at 100 Hz, 39 markers (Plug-in Gait full body); processed in Visual3D"
for key in order:
    r = key2row[key]
    sid = key
    g = int(r.GroupID)
    kk = 0
    for d in ("CW", "CCW"):
        h, p, t = (np.asarray(getattr(getattr(X, key), d), float) for X in (H, P, T))
        h, p, t = (x.reshape(x.shape[0], -1) for x in (h, p, t))
        ttm = np.atleast_1d(getattr(getattr(TT, key), d))
        tan = np.atleast_1d(getattr(getattr(TT, key).turnangle, d))
        stat["n_per_dir"][(key, d)] = p.shape[1]
        for j in range(p.shape[1]):
            ok = ~np.isnan(p[:, j])
            n = int(np.where(ok)[0][-1]) + 1  # padding at the end; a few NaN gaps can lie inside
            stat["pelvis_gaps"] = stat.get("pelvis_gaps", 0) + int((~ok[:n]).sum())
            kk += 1
            stat["turns"] += 1
            stat["nan_inside"] += int((np.isnan(h[:n, j]) & ok[:n]).sum() + (np.isnan(t[:n, j]) & ok[:n]).sum())
            fn = f"{key}_{d}{j + 1}.csv"
            tab = pd.DataFrame(dict(time_s=np.arange(n) / HZ, head_deg=h[:n, j], torso_deg=t[:n, j], pelvis_deg=p[:n, j]))
            tab.to_csv(TMP / "samples" / fn, index=False, lineterminator="\r\n", float_format="%.8g")
            extra.append((f"samples/{fn}", str(TMP / "samples" / fn)))
            ev = []
            try:
                ge = getattr(getattr(SC, key), f"{d}{j + 1}").gaitevents
                for nm, lab in [("Rheelstrike", "right_heel_strike"), ("Lheelstrike", "left_heel_strike"), ("Rheeloff", "right_heel_off"), ("Lheeloff", "left_heel_off")]:
                    for f in np.atleast_1d(getattr(ge, nm)):
                        ev.append(f"{lab}={(float(f) - 1) / HZ:.8g}")
                        stat["events_out"] += int(float(f) > n)
            except AttributeError:
                stat["short"] += 1
            tinfo = f"direction={'clockwise' if d == 'CW' else 'anticlockwise'};turn_in_direction={j + 1};turn_time_s={float(ttm[j]):.8g};turn_angle_deg={float(tan[j]):.8g}"
            rows.append(dict(
                dataset=ID, study="Leveridge et al. 2026 J Neuroeng Rehabil", experiment="", group=GROUP[g], subject=sid, person=sid,
                subject_excluded_by_authors=False, age=float(r.Age) if not np.isnan(r.Age) else np.nan, sex="", handedness="", population=POP[g],
                setting="lab", device=DEVICE, session=1, block=f"{'clockwise' if d == 'CW' else 'anticlockwise'} turns", trial=kk, phase="test",
                time_s=np.nan, task="walking", unit="recording", condition=f"360-degree turn on the spot, {'clockwise' if d == 'CW' else 'anticlockwise'}",
                pert_type="none", pert_value=0.0, pert_unit="", feedback="normal", excluded_by_authors=False, recording=kk, samples_file=f"samples/{fn}",
                signal_kind="transverse-plane (yaw) rotation angle of head, torso and pelvis in the laboratory frame, unwrapped",
                channels="head_deg;torso_deg;pelvis_deg", channel_units="deg;deg;deg", sample_rate_hz=HZ, n_samples=n, duration_s=(n - 1) / HZ,
                events=";".join(ev), processing=("Plug-in Gait segment angles from Visual3D, low-pass 4th-order Butterworth 6 Hz, unwrapped, cropped to start 75 frames before "
                                                 "the participant stopped in quiet stance on the target (authors' README); pelvis angle 0 is not re-zeroed here"),
                trial_info=tinfo))
df = pd.DataFrame(rows)

pc = [("group", "text", "", "Analysis group of the SPSS table: PD with freezing of gait, PD without freezing of gait, or healthy older adults."),
      ("age_years", "number", "years", "Age (Age); empty for one control."),
      ("gender_code", "integer", "1 or 2", "Gender as coded in the deposit (Gender); the deposit gives no value labels, so which code is women is not documented."),
      ("height_m", "number", "m", "Height."), ("weight_kg", "number", "kg", "Weight."),
      ("moca", "number", "points", "Montreal Cognitive Assessment (MOCA)."),
      ("hoehn_yahr", "number", "stage", "Hoehn and Yahr stage (HandY); Parkinson's groups only."),
      ("mini_bestest", "number", "points", "Mini-BESTest (MiniBEST); empty when missing."),
      ("mds_updrs_iii", "number", "points", "MDS-UPDRS part III (UPDRS); Parkinson's groups only."),
      ("mds_updrs_iii_without_fog_item", "number", "points", "MDS-UPDRS part III with the freezing item removed (UPDRS_NOFOGQ)."),
      ("steps_per_turn", "number", "steps", "Mean number of steps per 360-degree turn (TotalTurningSteps)."),
      ("turn_time_s", "number", "seconds", "Mean turn duration (turn_time)."),
      ("max_angular_difference_deg", "number", "deg", "Maximum head-pelvis angular difference over the turn (Max_ang_diff)."),
      ("avg_angular_difference_deg", "number", "deg", "Average head-pelvis angular difference over the turn (Avg_ang_diff)."),
      ("pelvis_latency_to_head", "number", "as in the deposit", "pelvis_latency_to_head; the deposit does not document it (values 0.04-0.2, probably seconds)."),
      ("max_start_deg", "number", "deg", "MAXstart: undocumented in the deposit; values 8-45 deg, probably the maximum head-pelvis angular difference in the start part of the turn."),
      ("max_middle_deg", "number", "deg", "MAXmiddle, as max_start_deg for the middle part."), ("max_end_deg", "number", "deg", "MAXend, as max_start_deg for the end part.")]
for part_, lab in [("START", "start"), ("MIDDLE", "middle"), ("LAST", "end")]:
    for nm, desc in [("In", "in-phase (head and pelvis rotating in the same direction)"), ("Anti", "anti-phase"), ("Distal", "distal-leading"), ("Proximal", "proximal-leading")]:
        pc.append((f"pct_{nm.lower()}_{lab}", "number", "% of stride", f"Percentage of the stride in the {desc} coupling-angle bin in the {lab} of the turn ({nm}{part_}); the bins follow the paper's vector coding (Fig. 2)."))
    pc.append((f"coordination_variability_{lab}_deg", "number", "deg", f"Circular standard deviation of the head-pelvis coupling angle across strides in the {lab} of the turn (CoVar_{ {'start': 'START', 'middle': 'MID', 'end': 'LAST'}[lab] })."))
part = []
for _, r in sp.iterrows():
    d_ = dict(person=r.key, group=GROUP[int(r.GroupID)], age_years=r.Age, gender_code=r.Gender, height_m=r.Height, weight_kg=r.Weight, moca=r.MOCA,
              hoehn_yahr=r.HandY, mini_bestest=r.MiniBEST, mds_updrs_iii=r.UPDRS, mds_updrs_iii_without_fog_item=r.UPDRS_NOFOGQ,
              steps_per_turn=r.TotalTurningSteps, turn_time_s=r.turn_time, max_angular_difference_deg=r.Max_ang_diff, avg_angular_difference_deg=r.Avg_ang_diff,
              pelvis_latency_to_head=r.pelvis_latency_to_head, max_start_deg=r.MAXstart, max_middle_deg=r.MAXmiddle, max_end_deg=r.MAXend)
    for part_, lab, cv in [("START", "start", "CoVar_START"), ("MIDDLE", "middle", "CoVar_MID"), ("LAST", "end", "CoVar_LAST")]:
        for nm in ("In", "Anti", "Distal", "Proximal"):
            d_[f"pct_{nm.lower()}_{lab}"] = r[f"{nm}{part_}"]
        d_[f"coordination_variability_{lab}_deg"] = r[cv]
    part.append(d_)
pdf = pd.DataFrame(part)

ck = stdlib.Checks()
gs = sp.groupby("GroupID").size().to_dict()
ck.published("participants with Parkinson's and freezing of gait", gs[1], 15, 0, "Results, Participant demographics")
ck.published("participants with Parkinson's without freezing of gait", gs[2], 14, 0, "Results, Participant demographics")
ck.published("healthy older adults", gs[3], 17, 0, "Results, Participant demographics")
ck.published("sampling rate (Hz)", 1 / np.median(np.diff(pd.read_csv(TMP / "samples" / extra[0][0].split("/")[1]).time_s)), 100, 0.01, "Methods (100 Hz motion capture)")


NOT_REPRO = []


def gm(col, g, f="mean"):
    v = sp[sp.GroupID == g][col]
    return v.mean() if f == "mean" else v.std()


for col, g, m_, s_, src in [("InSTART", 1, 92.3, 4.1, "start of turn, PD+FOG"), ("InSTART", 2, 95.1, 2.2, "start of turn, PD-FOG"), ("InSTART", 3, 94.3, 0.3, "start of turn, HC"),
                            ("InMIDDLE", 1, 93.4, 3.6, "middle of turn, PD+FOG"), ("InMIDDLE", 2, 95.4, 2.0, "middle of turn, PD-FOG"), ("InMIDDLE", 3, 96.5, 0.4, "middle of turn, HC"),
                            ("InLAST", 1, 83.9, 4.7, "end of turn, PD+FOG"), ("InLAST", 2, 86.8, 5.6, "end of turn, PD-FOG"), ("InLAST", 3, 83.5, 2.5, "end of turn, HC")]:
    for what, val, rep in [("mean", gm(col, g), m_), ("SD", gm(col, g, "sd"), s_)]:
        if abs(val - rep) <= 0.05:
            ck.published(f"percentage of the stride in-phase, {src}, {what}", val, rep, 0.05, "Results, En-bloc turning - percentage of the stride in-phase")
        else:
            NOT_REPRO.append(f"in-phase percentage, {src}, {what}: {val:.3f} here, {rep} in the paper")
ck.internal("46 participants have turn rows", df.subject.nunique() == 46)
print("empty pelvis samples inside turns:", stat.get("pelvis_gaps", 0), "head/torso empty where pelvis present:", stat["nan_inside"])
ck.internal("every heel event lies inside its recording", stat["events_out"] == 0, str(stat["events_out"]))
ck.internal("turn counts per direction: 5 for most, more or fewer for some", min(stat["n_per_dir"].values()) >= 3 and max(stat["n_per_dir"].values()) <= 9)

meta = dict(
    paper=dict(authors="Leveridge P, Russo Y, Williams GKR, Ye J, Wang Z, Lamb SE, Young WR", year=2026,
               title="Beyond en-bloc turning: dynamics of head-pelvis coordination in 360 deg turns in people with Parkinson's",
               venue="J NeuroEngineering Rehabil 23(1):221", doi="10.1186/s12984-026-01996-7"),
    lab="Young (University of Exeter)", data_url="https://osf.io/ydb4z/", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Every column of the three angle matrices (head, torso, pelvis) is one turn, written as a CSV of its non-empty samples (values unchanged to 8 significant digits; time is sample number / 100). "
             "Turn time, turn angle (turntime.mat) and the heel-strike and heel-off frames (step_count_curvature.mat) were added as trial_info and events; the SPSS table SPSS_main.sav is participants.csv. "
             "Not carried: the marker/insole file MOCAP_Insoles_Data.mat, the processed intermediate matrices (normalised strides, vector-coding outputs, SPM data) and the second SPSS file of most/least affected side."),
    how=["One row is one 360-degree turn on the spot (modified Ziegler protocol) of one participant: the unwrapped transverse-plane rotation of the head, torso and pelvis in degrees at 100 Hz, "
         "from 75 frames before the participant stopped in quiet stance on the target to the end of the turn. Participants walked between cones to a target, turned 360 degrees one way and then the other, "
         "about six targets clockwise and then about six anticlockwise; 3-9 turns per direction are in the deposit (turns with freezing of gait or without a clear stop before and after were removed by the authors before sharing).",
         "events are the authors' heel strikes and heel-offs of both feet (frame index of step_count_curvature.mat, converted to seconds from the start of the recording as (frame - 1) / 100, assuming MATLAB 1-based frames "
         "on the same time base as the angles; this is inferred). turn_time_s and turn_angle_deg in trial_info are the authors' turn duration and total rotation of the turn.",
         "The SPSS table is joined to the MATLAB participant keys through the mean turn time, which matches uniquely for all 46 participants. Groups follow the SPSS table (the analysis group); for 3 participants the MATLAB key "
         "prefix (PFOG or PNFOG) names the other PD group, and the SPSS group was used. sex is left empty because the deposit does not say which Gender code is women (gender_code is in participants.csv)."],
    unit_note="turn (one 360-degree turn: head, torso and pelvis rotation samples)",
    differences=[("Printed in the paper but not reproduced to its rounding (not used as checks): " + "; ".join(NOT_REPRO) + ".") if NOT_REPRO else "All printed in-phase percentages reproduced.",
                 f"{stat.get('pelvis_gaps', 0)} empty samples (NaN) lie inside the pelvis series of some turns (kept as empty cells); each series ends at its last non-empty sample.",
                 "The paper recruited 42 people with PD and 24 controls and says 8 PD+FOG, 5 PD-FOG and 2 controls were excluded; 15 + 14 + 17 = 46 analysed matches the deposit, but 24 - 2 = 22 controls does not equal 17 (as printed in the paper); the deposit holds the 17.",
                 "The pelvis and torso matrices hold one more participant (a PD-FOG participant with no head angles) than the head matrix; that participant is not in the 46 analysed and is not carried.",
                 "The deposit documents neither the Gender code nor MAXstart/MAXmiddle/MAXend and pelvis_latency_to_head; those columns are described as inferred.",
                 "Coordination variability and the other Table 3 and 4 values adjusted for covariates (with SEM) are not reproduced; only the unadjusted in-phase percentages of the Results text are used as checks."])
try:
    stdlib.release(ID, "timeseries", df, meta, ck, participants=pdf, participant_columns=pc, extra_files=extra)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
for c in ck.items:
    print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""), c.get("detail", ""))
