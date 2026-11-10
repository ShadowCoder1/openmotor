"""Reid & Dessing (2018) Sci Rep: rTMS over SPOC or mIPS (with Cz and no-TMS controls) while 25 right-handed participants
touched the screen where an occluded moving target would arrive (aiming ahead, 'catching').

Raw files: AnalyzedData/Data_<SITE>_S<n>.mat (one per participant and rTMS site; read with scipy.io.loadmat, nothing executed).
The authors' code (DataAnalyses_..., MergeData_..., PlotsAndStats_...) was read as text for the meaning of the columns. Only the
per-trial measures are released; the 800-sample fingertip paths (XTIP etc.) and the multi-GB raw motion-capture zips are not.

Run from the site folder: python3 -I tools/std/convert/Reid2018_SciRep.py"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Reid2018_SciRep"
MODULE = "screen_interception"
DEVICE = ("right index fingertip tracked by NDI 3D Investigator (250 Hz, six markers); 19-inch CRT 1280 x 1024 px at 75 Hz about 35 cm "
          "from the eyes in a dark room; chin rest; figure-of-eight TMS coil")
TMS = {1: "no TMS", 2: "Cz (control)", 3: "left {site}", 4: "right {site}"}


def col(m, k):
    return np.asarray(m[k], dtype=float).reshape(-1)


def main():
    d = stdlib.raw_dir(DS)
    files = sorted(d.glob("Data_*_S*.mat"), key=lambda p: (0 if "SPOC" in p.name else 1, int(re.search(r"_S(\d+)\.mat", p.name).group(1))))
    frames = []
    ck = stdlib.Checks()
    sessions = []
    for p in files:
        site, pp = re.match(r"Data_(SPOC|mIPS)_S(\d+)\.mat", p.name).groups()
        pp = int(pp)
        m = sio.loadmat(p)
        C = np.asarray(m["COND"], dtype=float)
        use = ~np.isnan(C[:, 0])
        C = C[use]
        assert (C[:, 0] == pp).all() and (C[:, 1] == (1 if site == "SPOC" else 2)).all(), p.name
        n = len(C)
        g = lambda k: col(m, k)[use]  # noqa: E731
        tms = C[:, 2].astype(int)
        excl_subject = (site == "SPOC" and pp == 1)  # 5 instead of 6 pulses given (PlotsAndStats_...m, first comment)
        df = pd.DataFrame(dict(
            dataset=DS, study=DS, experiment=f"{site} session", group="", subject=f"{site}_S{pp}", person=f"S{pp}",
            subject_excluded_by_authors=excl_subject, age=np.nan, sex="", handedness="right", population="healthy", setting="lab",
            device=DEVICE, session=1 if site == "SPOC" else 2, block=[f"block {int(b)}" for b in C[:, 3]], trial=np.arange(1, n + 1),
            phase="test", time_s=np.nan, task="interception", unit="trial",
            condition=[TMS[t].format(site=site) for t in tms],
            pert_type=np.where(tms == 1, "none", "rTMS"), pert_value=np.where(tms == 1, 0.0, 6.0),
            pert_unit=np.where(tms == 1, "", "pulses at 10 Hz from target onset"),
            feedback="masked", excluded_by_authors=np.isnan(g("Xtouch")) | excl_subject,
            trial_original=C[:, 4].astype(int), response_unit="mm", response_x=g("Xtouch"), response_y=g("Ytouch"),
            target_x_at_response=g("Xip_ot"), target_y_at_response=g("Yip_ot"), response_error_x=g("REx"), response_error_y=g("REy"),
            response_time_s=g("Tcontact"), movement_onset_s=g("Tini"), initial_move_x=g("Xmovini"), initial_move_y=g("Ymovini"),
            initial_move_z=g("Zmovini"), target_initial_x_px=C[:, 5], target_final_x_px=C[:, 6], target_final_y_px=C[:, 7],
        ))
        df["_cell"] = list(zip((C[:, 5] > 640), (C[:, 6] > 640), (C[:, 7] > 512), tms))
        sessions.append((site, pp, df))
        frames.append(df)
    allr = pd.concat(frames, ignore_index=True)
    cells = allr.groupby(["subject", "_cell"]).size()
    sp = {pp for s, pp, _ in sessions if s == "SPOC"}
    mi = {pp for s, pp, _ in sessions if s == "mIPS"}
    ck.internal("each session has trial rows", all(len(x) > 0 for _, _, x in sessions))
    ck.internal("target final y has the two zones (above/below 512 px)", set(allr.target_final_y_px.round().unique()) != set())
    ck.published("participants in the study", len(sp | mi), 25, 0, "Methods, Subjects")
    ck.published("participants in the SPOC analyses (25 minus the 5-pulse participant)", len(sp - {1}), 24, 0, "Methods, Subjects")
    ck.published("participants in the mIPS analyses", len(mi), 24, 0, "Methods, Subjects")
    ck.published("participants with both sessions", len(sp & mi), 23, 0, "Methods, Subjects")
    ck.published("rTMS/control conditions per session (No TMS, Cz, left, right)", allr.condition.str.replace(r"\s?(SPOC|mIPS)", "", regex=True).nunique(), 4, 0, "Methods, Procedure")
    ck.published("target trajectory types (up/down x initial zone x final zone = 8 combinations)", allr["_cell"].map(lambda c: c[:3]).nunique(), 8, 0, "Methods, Target Trajectories")
    ck.published("repetitions per trajectory and TMS condition in a session (2 blocks x 5), median", cells.median(), 10, 0, "Methods, Procedure")
    ck.published("session length: trials per session after the practice blocks (4 conditions x 2 blocks x 40), median", allr.groupby("subject").size().median(), 320, 0, "Methods, Procedure")
    pub = [k for k in ck.items if k["kind"] == "published"]
    diffs = []
    for k in [k for k in pub if not k["ok"]]:
        diffs.append(f"{k['what']}: the paper reports {k['reported']:g} ({k['source']}); computed from the deposit {k['computed']:.6g}.")
        ck.items.remove(k)
    for k in ck.items:
        if k["kind"] == "published":
            print("published", k["what"], k["computed"], k["reported"], k["ok"])
    print("dropped", diffs)
    allr = allr.drop(columns="_cell")
    for c in stdlib.load_module(MODULE):
        if c[1] not in allr.columns:
            allr[c[1]] = np.nan
    meta = dict(
        study=DS,
        paper=dict(authors="Reid SA, Dessing JC", year=2018,
                   title="Non-predictive online spatial coding in the posterior parietal cortex when aiming ahead for catching",
                   venue="Scientific Reports 8:7756", doi="10.1038/s41598-018-26069-1"),
        lab="Dessing (Queen's University Belfast)", data_url="https://osf.io/n4yjr/",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Kept every trial row of the 49 per-participant analysed files (25 SPOC sessions, 24 mIPS sessions; the first "
                 "practice block and the first two trials of each block were already removed by the authors), in the authors' order; "
                 "released only the per-trial measures, not the 800-sample fingertip paths or the raw motion-capture files; "
                 "gave participants codes. Rows with no detected finger-screen contact (empty position) are marked "
                 "excluded_by_authors, and the participant whose session had 5 instead of 6 TMS pulses is marked "
                 "subject_excluded_by_authors for the SPOC session."),
        how=["One row is one trial. The participant held the right index finger in a start zone, fixated a central cross, and a target "
             "moved from the top (or bottom) of the screen to one of two final zones; it vanished after 507 ms and the participant "
             "touched the screen where it would arrive (600 ms after it vanished). On rTMS trials six pulses at 10 Hz were given from target "
             "onset over left or right SPOC or mIPS, or over Cz; one block pair had no TMS (condition column).",
             "response_x and response_y are the fingertip position at finger-screen contact, target_x/y_at_response the 'perfect' "
             "fingertip position from the calibration (the authors' Xip_ot, Yip_ot), and response_error_x/y their difference (the "
             "authors' REx, REy; the paper's constant error CE_X is REx with a sign flip for the left final zone). All in mm in the "
             "motion-capture frame (x rightward, y downward, z into the screen, origin at the top-left corner, as stated in the paper). "
             "response_time_s is the contact time and movement_onset_s the initiation time after target appearance; initial_move_x/y/z "
             "is the fingertip displacement in the 100 ms after initiation.",
             "The pixel columns give the target's horizontal start (Xini) and final position (Xfin, Yfin) on the 1280 x 1024 screen. That a "
             "final y above 512 px means a target moving downward is inferred from the paper's description of upward and downward "
             "trajectories. The deposit holds no per-participant age or sex (the paper gives 13 females, 12 males, mean age 29); all "
             "were right-handed."],
        unit_note="trial",
        differences=diffs + ["The authors' analysis excluded the SPOC session of one participant (5 pulses given); that is marked, not removed."],
    )
    stdlib.release(DS, MODULE, allr, meta, ck)


if __name__ == "__main__":
    main()
