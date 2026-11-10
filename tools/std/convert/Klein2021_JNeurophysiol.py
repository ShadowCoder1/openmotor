"""Klein, Maiello, Fleming & Voudouris 2021 (J Neurophysiol): 21 adults grasped a small object with thumb and index finger; thumb and
index positions were tracked (Optotrak). Object surfaces (brass, wood, sandpaper) and the object's orientation (aligned with the
participant's natural grasp axis, NGA, or not) varied. Deposit (Zenodo 4534892): Data_and_Analyses.zip with one .mat per participant
(100 grasps with 500 samples x 6 finger coordinates + 9 table-marker coordinates, plus a separate file of 10 NGA grasps)."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Klein2021_JNeurophysiol"
MODULE = "timeseries"
BASE = "Data_and_Analyses/"
CH = ["thumb_x", "thumb_y", "thumb_z", "index_x", "index_y", "index_z"]
CHT = [f"table_marker{k}_{a}" for k in (1, 2, 3) for a in "xyz"]
OBJ = {1: "wood/brass", 2: "brass/brass", 3: "sandpaper/brasseline"}
ORI = {1: "wood-sandpaper side clockwise", 2: "brass-brasseline side clockwise", 3: "brass-brass"}


def cond_id(angle, ori, obj):
    a = 0 if abs(angle - 22.5) < 1e-6 else 5 if abs(angle - 45) < 1e-6 else None
    tab = {(3, 2): 1, (1, 1): 2, (2, 1): 3, (1, 3): 4, (2, 3): 5}  # (orientation, object) -> offset, as in Parse_Data.m
    tab = {(3, 2): 1, (1, 1): 2, (2, 1): 3, (1, 3): 4, (2, 3): 5}
    return a + tab[(int(ori), int(obj))]


def surf(c):
    return {1: "brass on both sides", 2: "wood on clockwise side", 3: "wood on counterclockwise side",
            4: "sandpaper on clockwise side", 5: "sandpaper on counterclockwise side"}[(c - 1) % 5 + 1]


def ld(z, name):
    return sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True, struct_as_record=False)


def sample_csv(tdi):
    v = np.asarray(tdi.Variables, float)
    t = np.asarray(tdi.Tablemarkers, float)
    n = len(v)
    assert t.shape[0] == n and v.shape[1] == 6 and t.shape[1] == 9
    a = np.column_stack([np.arange(n) / 100.0, v, t])
    df = pd.DataFrame(a, columns=["time_s"] + CH + CHT)
    return df.to_csv(index=False, float_format="%.8g", lineterminator="\r\n").encode(), n


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(next(stdlib.raw_dir(DS).glob("*.zip")))
    dnames = sorted(n for n in z.namelist() if n.startswith(BASE + "Data/S") and n.endswith(".mat"))
    nnames = {Path(n).name[:3]: n for n in z.namelist() if n.startswith(BASE + "Data/NGAData/S") and n.endswith(".mat")}
    parsed = ld(z, BASE + "ParseData/Parsed_Data.mat")
    rows, extra, persons, ages, sexes, hands, ntr, nnga, perm_ok, cond_counts = [], [], [], [], [], [], [], [], [], []
    for dn in dnames:
        sid = Path(dn).name[:3]
        m = ld(z, dn)
        si = m["SubjectInfo"]
        n_ng = ld(z, nnames[sid])
        sex = {"M": "M", "F": "F"}.get(str(si.gender).strip().upper()[:1], "other")
        hand = {"R": "right", "L": "left"}.get(str(si.handedness).strip().upper()[:1], "ambi")
        ages.append(int(si.age)); sexes.append(sex); hands.append(hand)
        base = dict(dataset=DS, study=DS, experiment="", group="", subject=sid, person=sid, subject_excluded_by_authors=False,
                    age=int(si.age), sex=sex, handedness=hand, population="healthy", setting="lab",
                    device="Optotrak motion capture, markers on thumb and index finger", session=1, time_s="",
                    task="grasping and lifting", unit="recording", pert_type="none", pert_value=0, pert_unit="",
                    feedback="normal", excluded_by_authors=False, signal_kind="3D positions of thumb and index finger markers and 3 table markers",
                    channels=";".join(CH + CHT), channel_units=";".join(["mm"] * 15), sample_rate_hz=100,
                    processing="none (raw tracked positions; the authors' gap handling and resampling are in their Parse_Data.m)")
        rec = 0
        for k, td in enumerate(n_ng["TrialData"], 1):
            rec += 1
            csvb, n = sample_csv(td)
            f = f"samples/{sid}_NGA_{k:02d}.csv"
            extra.append((f, csvb))
            rows.append({**base, "block": "NGA (natural grasp axis)", "trial": rec, "phase": "baseline", "condition": "natural grasp axis trial",
                         "recording": rec, "samples_file": f, "n_samples": n, "duration_s": round((n - 1) / 100, 6), "events": "",
                         "trial_info": "trial_type=NGA"})
        nnga.append(len(n_ng["TrialData"]))
        angle, ori, obj = np.asarray(si.Angle, float), np.asarray(si.Orientation), np.asarray(si.Objects)
        order = np.asarray(si.ShuffledTrials).astype(int)
        perm_ok.append(sorted(order) == list(range(1, len(order) + 1)))
        conds = np.array([cond_id(a, o, b) for a, o, b in zip(angle, ori, obj)])
        cond_counts.append(np.bincount(conds, minlength=11)[1:])
        tds = list(m["TrialData"])
        assert len(tds) == len(order) == 100
        for t in np.argsort(order, kind="stable"):
            rec += 1
            csvb, n = sample_csv(tds[t])
            f = f"samples/{sid}_trial{order[t]:03d}.csv"
            extra.append((f, csvb))
            c = int(conds[t])
            rows.append({**base, "block": "main", "trial": rec, "phase": "test",
                         "condition": f"{'NGA-aligned (45 deg)' if angle[t] == 45 else 'NGA-unaligned (22.5 deg)'}, {surf(c)}",
                         "recording": rec, "samples_file": f, "n_samples": n, "duration_s": round((n - 1) / 100, 6), "events": "",
                         "trial_info": f"condition_id={c};object_id={int(obj[t])};object={OBJ[int(obj[t])]};orientation_id={int(ori[t])};"
                                       f"orientation={ORI[int(ori[t])]};angle_deg={angle[t]:g};presentation_order={int(order[t])}"})
        ntr.append(len(tds))
        p = dict(person=sid)
        rd = m["RatingData"]
        assert len(rd) == 4
        fr, pl = {}, {}
        for r in rd:
            fc, fv = np.atleast_1d(r.FrictionCondition), np.atleast_1d(r.Friction)
            pc, pv = np.atleast_1d(r.PleasantnessCondition), np.atleast_1d(r.Pleasantness)
            for a_, b_ in zip(fc, fv): fr[int(a_)] = float(b_)
            for a_, b_ in zip(pc, pv): pl[int(a_)] = float(b_)
        for i in range(1, 5):
            p[f"friction_coeff_{i}"] = fr[i]; p[f"pleasantness_{i}"] = pl[i]
        persons.append(p)
    df = pd.DataFrame(rows)
    n = len(dnames)
    ages = np.array(ages)
    pa = np.asarray(parsed["Age"]).astype(float)
    ck.internal("ages equal the authors' Parsed_Data ages", np.array_equal(np.sort(pa), np.sort(ages)))
    ck.internal("100 grasps and 10 NGA grasps per participant", set(ntr) == {100} and set(nnga) == {10})
    ck.internal("ShuffledTrials is a permutation of 1..100 for every participant", all(perm_ok))
    ck.internal("each of the 10 conditions has 10 grasps per participant", all((c == 10).all() for c in cond_counts))
    ck.internal("row count", len(df) == 21 * 110 and len(extra) == len(df), str(len(df)))
    ck.published("participants (21)", n, 21, 0, "Results, Participants (Stats_Results.txt: N = 21)")
    ck.published("mean age, years (24.4)", ages.mean(), 24.4, 0.05, "Methods, Participants (deposit Stats_Results.txt)")
    ck.published("youngest age (18)", ages.min(), 18, 0, "Methods, Participants")
    ck.published("oldest age (35)", ages.max(), 35, 0, "Methods, Participants")
    ck.published("females (16)", sum(s == "F" for s in sexes), 16, 0, "Methods, Participants")
    ck.published("right-handed (21)", sum(h == "right" for h in hands), 21, 0, "Methods, Participants")
    meta = dict(
        study=DS,
        paper=dict(authors="Klein LK, Maiello G, Fleming RW, Voudouris D", year=2021,
                   title="Friction is preferred over grasp configuration in precision grip grasping",
                   venue="J Neurophysiol 125(4):1330-1338", doi="10.1152/jn.00021.2021"),
        lab="Fleming (Justus Liebig University Giessen)", data_url="https://doi.org/10.5281/zenodo.4534892",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Wrote each grasp's tracked thumb and index finger positions and the three table-marker positions to a CSV (time_s plus 15 channels, "
                 "same values); put the 10 natural-grasp-axis (NGA) grasps before the 100 main grasps of each participant, and the main grasps in the "
                 "order they were presented (the deposit's ShuffledTrials); named conditions from the authors' angle, orientation and object codes with "
                 "the mapping in their Parse_Data.m; dropped the date-stamped file names and the unreadable MATLAB transform object; participant IDs became codes."),
        how=["One row is one grasp (one recording) of one participant: a thumb and index finger grasp of a small object, with 500 samples (a few have 501) "
             "of thumb and index x, y, z and three table-marker x, y, z positions, in mm (Optotrak), at 100 Hz. The 100 main grasps are 10 repeats of "
             "each of 10 conditions (object angle 22.5 deg = NGA-unaligned or 45 deg = NGA-aligned, times five surface set-ups: brass, or wood or sandpaper on the clockwise or counterclockwise side). The 10 NGA grasps come first.",
             "The sampling rate of 100 Hz is the rate used by the authors' Parse_Data.m; the deposit's SubjectInfo has a FrameRate field of 50 and 200 frames per trial, which does not match the 500 stored samples. "
             "Within trial_info, condition_id is the authors' 1-10 coding (1-5 unaligned, 6-10 aligned). The main trials are sorted by the presentation order; the 6 practice trials are not in the deposit.",
             "participants.csv holds each participant's friction coefficients and pleasantness ratings for the four rated surface set-ups, numbered 1-4 as in the deposit (the deposit does not name the set-ups)."],
        unit_note="grasp (recording of about 5 s)",
        differences=["The authors' clockwise/counterclockwise grip classification and the mean NGA of 66 deg need their contact-detection routine (MSI.m) with resampling; it was not re-run, so these published numbers were not reproduced here. The participant counts and demographics were."],
    )
    pdf = pd.DataFrame(persons)
    pcols = [(f"friction_coeff_{i}", "number", "", f"Measured friction coefficient of rated surface set-up {i} (numbered as in the deposit).") for i in range(1, 5)] + \
            [(f"pleasantness_{i}", "number", "", f"Rated pleasantness of surface set-up {i} (numbered as in the deposit; scale as in the paper).") for i in range(1, 5)]
    stdlib.release(DS, MODULE, df, meta, ck, participants=pdf, participant_columns=pcols, extra_files=extra)


if __name__ == "__main__":
    main()
