"""LeBlanc, Lacroix, Benwell, Grant, Feltmate & Neyedli (2026), bioRxiv 10.64898/2026.10.08.757378, Grasping Under Target Size and Location Uncertainty:
go-before-you-know (GBYK) and know-before-you-go (KBYG) precision grasps of one of two cylinders with OptiTrack markers on the right hand.
Borealis doi 10.5683/SP4/5OITGH (CC0): Grasp_T_UnC_data.tab (grip aperture and hand position, time-normalized to 1000 samples per trial) and
ref_time.tab (per-trial parameters and times). Only trials the authors kept are in the deposit.
Run on the PC: python3 -I tools/std/pc.py run LeBlanc2026_Borealis"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "LeBlanc2026_Borealis"
B = stdlib.raw_dir(DS)
TMP = Path("tmp_" + DS)
shutil.rmtree(TMP, ignore_errors=True)
(TMP / "samples").mkdir(parents=True)
KEY = ["version", "pID", "cond", "trial"]
EXP = {"closed": "Experiment 1 (fingers together at start)", "open": "Experiment 2 (fingers 10.5 cm apart at start)"}
CHANNELS = ["movement_pct", "grip_aperture_mm", "x_hand_mm", "y_hand_mm", "z_hand_mm"]


def main():
    ck = stdlib.Checks()
    ref = pd.read_csv(B / "ref_time.tab", sep="\t")
    big = pd.read_csv(B / "Grasp_T_UnC_data.tab", sep="\t", dtype={"version": "category", "cond": "category", "t_size": "category", "d_size": "category", "t_loc": "category"})
    ref["_k"] = list(zip(ref.version, ref.pID, ref.cond, ref.trial))
    refi = ref.set_index("_k")
    assert refi.index.is_unique
    cond_order = {"GBYK": 0, "KBYG": 1}
    ref["_o"] = ref.cond.map(cond_order)
    ref = ref.sort_values(["version", "pID", "_o", "trial"], ascending=[False, True, True, True]).reset_index(drop=True)  # closed (Exp 1) first
    groups = {k: g for k, g in big.groupby(KEY, sort=False, observed=True)}
    ck.internal("every trial of ref_time has a 1000-sample series and vice versa", len(groups) == len(ref) and set(groups) == set(refi.index))
    rows, extra = [], []
    nsubj = {}
    rec = {}
    n1000 = True
    mism = 0
    for _, r in ref.iterrows():
        k = (r.version, r.pID, r.cond, r.trial)
        g = groups[k].sort_values("index")
        n1000 &= len(g) == 1000
        mism += int(not ((g.t_size == r.t_size).all() and (g.d_size == r.d_size).all() and (g.t_loc == r.t_loc).all()))
        sk = (r.version, r.pID)
        if sk not in nsubj:
            nsubj[sk] = len(nsubj) + 1
        rec[sk] = rec.get(sk, 0) + 1
        fn = f"samples/p{nsubj[sk]:03d}_r{rec[sk]:03d}.csv"
        t = (g["index"].to_numpy() / 100.0) * float(r.mt)
        arr = np.column_stack([t, g["index"].to_numpy(), g.grip_apt.to_numpy(), g.x_hand.to_numpy(), g.y_hand.to_numpy(), g.z_hand.to_numpy()])
        with open(TMP / fn, "w", newline="") as f:
            f.write("time_s," + ",".join(CHANNELS) + "\r\n")
            np.savetxt(f, arr, fmt="%.8g", delimiter=",", newline="\r\n")
        extra.append((fn, str(TMP / fn)))
        ti = (f"target_size={r.t_size};distractor_size={r.d_size};target_location={r.t_loc};object_grasped={r.object};reaction_time_s={r.rt:.8g};"
              f"movement_time_s={r.mt:.8g};go_signal_s={r.go_sign:.8g};movement_start_s={r.start:.8g};target_onset_pct_of_movement={r.ptarget:.8g};distance_threshold_mm={int(r.threshold)};authors_trial={int(r.trial)}")
        rows.append(dict(dataset=DS, study="LeBlanc 2026 bioRxiv", experiment=EXP[r.version], group="", subject=f"{r.version}_{r.pID}", person=f"{r.version}_{r.pID}",
                         subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                         device="OptiTrack 3D motion capture (8 PrimeX 13 cameras, 120 Hz), markers on the right hand; two cylinders (4 cm and 8 cm diameter) on a monitor",
                         session=1, block=f"{r.cond} condition", trial=rec[sk], phase="test", time_s="", task="grasping and lifting", unit="recording", condition=r.cond,
                         pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                         recording=rec[sk], samples_file=fn, signal_kind="reach-to-grasp: grip aperture (thumb-index marker distance) and mean hand-marker position, time-normalized to movement time",
                         channels=";".join(CHANNELS), channel_units="% of movement time;mm;mm;mm;mm", sample_rate_hz="", n_samples=len(g), duration_s=float(r.mt),
                         events=f"target_onset_pct_of_movement={r.ptarget:.8g}",
                         processing="4th-order dual-pass Butterworth low-pass 6 Hz; hand position recentred on the initial hand position; spline-normalized to 1000 samples per trial between releasing the start key and the start of lifting the object",
                         trial_info=ti))
    df = pd.DataFrame(rows)
    ntr = ref.groupby(["version", "pID"]).size()
    ck.internal("1000 samples in every trial", bool(n1000))
    ck.internal("target/distractor sizes and location agree between the two files", mism == 0)
    ck.internal("recordings = rows of ref_time", len(df) == len(ref) == 7694)
    ck.published("participants in Experiment 2 (25)", ref[ref.version == "open"].pID.nunique(), 25, 0, "Experiment 2, Participants")
    ck.published("test trials per condition numbered up to 80", ref.trial.max(), 80, 0, "Materials and methods, Experiment 1, Procedure (80 test trials per condition)")
    ck.internal("no participant has more than 160 kept trials (2 conditions x 80 test trials)", bool(ntr.max() <= 160), str(ntr.max()))
    cmb = ref.groupby(["version", "pID", "cond", "t_size", "d_size", "t_loc"]).size()
    ck.published("most trials of one target size x distractor size x location combination (10)", cmb.max(), 10, 0, "Procedure (10 trials per combination)")
    ck.published("combinations of target size, distractor size and target location (8)", ref.groupby(["t_size", "d_size", "t_loc"]).ngroups, 8, 0, "Procedure")
    n_part = ref.groupby(["version", "pID"]).ngroups
    ck.published("percent of trials removed, all participants x 160 (5.7)", 100 * (1 - len(ref) / (n_part * 160)), 5.7, 0.05, "Experiment 1, Data Processing (5.7% of trials removed)")
    ck.internal("both conditions and both experiments present", df.condition.nunique() == 2 and df.experiment.nunique() == 2)
    meta = dict(study=DS, paper=dict(authors="LeBlanc KA, Lacroix ATD, Benwell JA, Grant M, Feltmate BBT, Neyedli HF", year=2026, title="Grasping Under Target Size and Location Uncertainty",
                venue="bioRxiv (preprint, not yet published as of 2026-11-09)", doi="10.64898/2026.10.08.757378"),
        lab="Neyedli (Dalhousie University)", data_url="https://doi.org/10.5683/SP4/5OITGH", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("One row per trial (7,694 trials) from ref_time.tab; the deposit's 1000-sample series of each trial (Grasp_T_UnC_data.tab, 7.7 million rows) is written as one CSV per trial in samples/. "
                 "Columns were renamed with units; the authors' percent-of-movement index (0.1 to 100) is kept as movement_pct and time_s was added as movement_pct/100 x the trial's movement time (seconds). "
                 "Target/distractor sizes and locations, object grasped, reaction time, movement time, go-signal time, movement start, target onset and the GBYK distance threshold are in trial_info. "
                 "Participant numbers became codes. Practice trials and the trials the authors removed are not in the deposit."),
        how=["One row is one grasp: the participant reaches from the start key to one of two cylinders (small 4 cm or large 8 cm) and lifts the one that is the target. In GBYK the target was shown only after the hand had moved 5-10 cm (distance_threshold_mm), in KBYG before the go tone (distance_threshold_mm 0, target_onset 0).",
             "samples/<file>.csv holds 1000 samples: movement_pct, grip_aperture_mm (distance between thumb and index markers), x_hand_mm, y_hand_mm, z_hand_mm (mean of the hand markers, origin at the initial hand position, mm, axes as the authors' recentring: x and y shifted, z flipped). The movement runs from releasing the start key to the start of lifting.",
             "Times in trial_info are seconds (the deposit's codebook says ms, but its processing code and values are in seconds: reaction time 0.29 s on average, movement time 0.52 s). target_onset_pct_of_movement can exceed 100 when the target appeared after the end of the movement. The codebook's description of grip_apt as a y position is a typo; it is the aperture in mm."],
        unit_note="grasp (one trial, with a samples CSV)",
        differences=["The paper reports 23 participants in Experiment 1; the deposit holds trials of 26 participants in Experiment 1 (three more than reported), and 25 in Experiment 2 as reported. The extra three participants were not identified in the deposit and are kept.",
                     "The paper's 5.7% of trials removed is given under Experiment 1; it reproduces over both experiments combined (51 participants x 160 trials), not for Experiment 1 alone (5.0% with 26 participants).",
                     "Age, sex and handedness are not in the deposit (ref_demographic.csv is mentioned in its README but not shared); the paper gives 17 of 23 and 17 of 25 women."])
    stdlib.release(DS, "timeseries", df, meta, ck, extra_files=extra)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
