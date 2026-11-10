"""Lepori, Hartmann, Dehn, Chessa, Fleming & Maiello 2025 (J Neurophysiol): do findings about precision grips hold when people grasp
freely with any number of digits? 20 right-handed adults grasped four 3D polycube objects (two shapes, two material configurations of
wood and brass cubes, 716 g) five times each in an unconstrained session (first) and a precision-grip session (second), with a Qualisys
motion-capture system (24 markers on the hand, 4 on the object).

Deposit (Zenodo 14973756): Scripts_and_Data.zip (158 MB): the raw 3D/6D recordings (tsv) and MATLAB scripts of the pipeline, plus the
authors' derived per-trial files. This converter reads the derived per-trial values only (nothing is executed; scripts were read as text):
 - Analysis and Figures/Number of Fingers/finger_count/finger_count_{prec,unc}_subjNN.txt (20 values: digits in contact per trial)
 - Preprocessing/{Precision,Unconstrained}/.../subjNN_{prec,unc}/subjNN_*_contact_times.mat (contact_times 5 x 20 digits x trials, first_contact 20)
 - Analysis and Figures/Objects Tilt/{Precision,Unconstrained}/{SB,SW,VB,VW}/XX_{prec,unc}_subjNN_tilt.mat (5 values per object)
The 20 trial slots of every file are ordered by object as in the authors' scripts (slots 1-5 VW, 6-10 VB, 11-15 SB, 16-20 SW), with the five
repetitions of an object in the authors' file order (the raw folders contain renamed and duplicated copies of some recordings, so the slot
order is not the order of the trials in time). The raw marker recordings are not converted. The signed centroid-distance files are not
converted either: their order within the 10 values of each material configuration is not documented.

Run from the site folder: python3 -I tools/std/convert/Lepori2025_JNeurophysiol.py"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Lepori2025_JNeurophysiol"
MODULE = "multidigit_grasp"
RAW = stdlib.raw_dir(DS)
ROOT = "Scripts_and_Data/"
OBJ = ["VW", "VB", "SB", "SW"]
DIGITS = ["thumb", "index", "middle", "ring", "little"]
DEVICE = "Qualisys motion capture (8 Miqus M5 cameras, 100 Hz), 24 markers on the hand, 4 on the object"
SESS = [("unc", "Unconstrained", "unconstrained grasp", 1), ("prec", "Precision", "precision grip", 2)]


def f(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(RAW / "Scripts_and_Data.zip")
    rows = []
    fc_all, ct_all = {}, {}
    for key, folder, label, snum in SESS:
        fcs, cts = [], []
        for p in range(1, 21):
            fc = np.array([float(x) for x in z.read(f"{ROOT}Analysis and Figures/Number of Fingers/finger_count/finger_count_{key}_subj{p:02d}.txt").decode().strip().split(",")])
            fold = "Precision/Subjects_precision" if key == "prec" else "Unconstrained/Subjects_unconstrained"
            m = sio.loadmat(io.BytesIO(z.read(f"{ROOT}Preprocessing/{fold}/subj{p:02d}_{key}/subj{p:02d}_{key}_contact_times.mat")), squeeze_me=True)
            ct, first = np.asarray(m["contact_times"], float), np.asarray(m["first_contact"], float)
            assert fc.shape == (20,) and ct.shape == (5, 20) and first.shape == (20,)
            tilt = []
            for o in OBJ:
                t = sio.loadmat(io.BytesIO(z.read(f"{ROOT}Analysis and Figures/Objects Tilt/{folder}/{o}/{o}_{key}_subj{p:02d}_tilt.mat")), squeeze_me=True)
                tilt.append(np.atleast_1d(np.asarray(t[f"tilt_{o}"], float)))
            tilt = np.concatenate(tilt)
            assert tilt.shape == (20,)
            rel = (ct - (first + 15)[None, :]) / 100.0  # seconds relative to the object starting to move (frame first_contact + 15, 100 Hz)
            fcs.append(fc)
            cts.append(rel)
            for slot in range(20):
                rows.append(dict(
                    dataset=DS, study=DS, experiment="Precision grip vs unconstrained grasping", group="", subject=f"P{p:02d}", person=f"P{p:02d}",
                    subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy", setting="lab", device=DEVICE,
                    session=snum, block=label, trial=0, phase="test", time_s="", task="grasping and lifting", unit="trial",
                    condition=f"{label}, object {OBJ[slot // 5]}", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                    excluded_by_authors=bool(first[slot] == 1),
                    object_code=OBJ[slot // 5], repetition=slot % 5 + 1, n_digits_contact=int(fc[slot]), first_contact_frame=int(first[slot]),
                    **{f"contact_time_{d}_s": f(rel[i, slot]) for i, d in enumerate(DIGITS)}, max_object_tilt_deg=f(tilt[slot])))
        fc_all[key], ct_all[key] = np.array(fcs), np.array(cts)
    # trial number = position in the participant's sequence
    cnt = {}
    for r in rows:
        cnt[r["subject"]] = cnt.get(r["subject"], 0) + 1
        r["trial"] = cnt[r["subject"]]
    A, U = fc_all["prec"], fc_all["unc"]
    W, B = [0, 1, 2, 3, 4, 15, 16, 17, 18, 19], list(range(5, 15))  # the authors' 'W' and 'B' groups of slots (objects VW, SW and VB, SB)
    cell = {("u", "B"): U[:, B].mean(1), ("u", "W"): U[:, W].mean(1), ("p", "B"): A[:, B].mean(1), ("p", "W"): A[:, W].mean(1)}
    Fg = stats.ttest_1samp((cell["u", "B"] + cell["u", "W"]) / 2 - (cell["p", "B"] + cell["p", "W"]) / 2, 0).statistic ** 2
    Fm = stats.ttest_1samp((cell["u", "B"] + cell["p", "B"]) / 2 - (cell["u", "W"] + cell["p", "W"]) / 2, 0).statistic ** 2
    Fi = stats.ttest_1samp((cell["u", "B"] - cell["u", "W"]) - (cell["p", "B"] - cell["p", "W"]), 0).statistic ** 2
    ck.published("participants (20)", len({r["subject"] for r in rows}), 20, 0, "Methods, Participants; the ANOVAs' 19 error degrees of freedom")
    ck.published("trials per session (20 = 5 repetitions x 4 objects)", len(rows) / 20 / 2, 20, 0, "Methods, Procedure")
    ck.published("number of digits, effect of grasp type, F(1,19) = 196.90", Fg, 196.90, 0.05, "Results, Number of digits (2 grasp types x 2 material configurations, subjects' means)")
    ck.published("number of digits, effect of material configuration, F(1,19) = 1.16", Fm, 1.16, 0.01, "Results, Number of digits")
    ck.published("number of digits, interaction, F(1,19) = 0.18", Fi, 0.18, 0.01, "Results, Number of digits")
    M = np.nanmean(ct_all["prec"], axis=2)  # participants x digits
    d = M[:, 0] - M[:, 1]
    d = d[~np.isnan(d)]
    r = stats.rankdata(np.abs(d))
    n = len(d)
    zv = (r[d > 0].sum() - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    ck.published("precision grip: thumb vs index contact time, signed-rank z = 2.05", abs(zv), 2.05, 0.01, "Results, Contact times (Wilcoxon signed-rank test of the 20 participants' mean contact times)")
    ck.internal("800 rows = 20 participants x 2 sessions x 20 trials", len(rows) == 800)
    ck.internal("digits in contact are between 0 and 5", bool(((A >= 0) & (A <= 5)).all() and ((U >= 0) & (U <= 5)).all()))
    nfl = sum(1 for r in rows if r["excluded_by_authors"])
    ck.internal(f"trials the authors flagged as mislabelled: {nfl}", True)
    meta = dict(
        study=DS,
        paper=dict(authors="Lepori F, Hartmann FT, Dehn KI, Chessa M, Fleming RW, Maiello G", year=2025,
                   title="Does precision grip research extend to unconstrained, multidigit grasping?",
                   venue="J Neurophysiol 133(6):1836-1843", doi="10.1152/jn.00008.2025"),
        lab="Maiello (University of Southampton) and Fleming (Justus Liebig University Giessen)", data_url="https://doi.org/10.5281/zenodo.14973756",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Only the authors' derived per-trial files were read out of the 158 MB ZIP: digits in contact, contact time of each digit, the frame taken as first contact, and the maximum tilt of the object. "
                 "Contact times were converted from frames to seconds relative to the object starting to move, as in the authors' analysis script. Participant numbers became codes; session 1 is the unconstrained session and session 2 the precision-grip session, as in the paper. "
                 "Not included: the raw 3D marker and 6D object recordings (about 580 MB of text), the authors' signed distances of the grasp centroid from the object centroid and the grasp-variability values (their order within the files is not documented), and the object meshes and MATLAB scripts."),
        how=["One row is one grasp: one participant grasping one of four 3D objects (codes VW, VB, SB, SW: shape V or S, and one of two arrangements W or B of the five wooden and five brass cubes) in one session, in the order of the authors' data slots (objects in the order VW, VB, SB, SW with five repetitions each). "
             "The slot order within an object is the order of the recordings' files, not necessarily the order in time, so trial counts the row within the participant's two sessions.",
             "n_digits_contact is the number of the five digits whose fingertip marker was within 25 mm of the object at first contact. Contact times are the frames at which each digit's marker was closest to the object, minus the frame at which the object started moving (first contact + 15 frames), divided by 100 (100 Hz); they are empty for digits that were not in contact. "
             "In the precision-grip session only thumb and index finger were meant to touch. max_object_tilt_deg is the maximum over the trial of sqrt(roll^2 + pitch^2) of the object from the 6D data (the authors' tilt analysis), empty where the authors set it to missing.",
             "excluded_by_authors marks trials whose first-contact frame the authors' script sets to 1 ('labeling error' in their hand corrections of the participant scripts); the tilt of those is missing in the deposit."],
        unit_note="grasp (one object, one repetition)",
        differences=["The authors' scripts set the first-contact frame to 1 for 24 of the 800 trials (3%) and then set their tilt to missing; the paper says 2% of trials were excluded and 7% re-processed by hand, so these 24 may include trials the paper counts differently.",
                     "The paper's text defines contact time as the time a digit reached minimum distance to the object relative to object motion, which is how it is computed here; the paper's unconstrained-session thumb-versus-index test (z = 3.92) was not reproduced (z = 1.64 from the 20 participants' mean contact times, thumb minus index), so its definition may differ from the authors' script.",
                     "The paper gives no age, sex or handedness per participant (20 right-handed, ages 20-39, mean 26.7, 15 women); none are in the table.",
                     "Raw recordings contain renamed and duplicated copies of some trial files (for example '- Copia'), which the authors' scripts read by sorted file order; the derived values are released as the authors computed them."],
    )
    stdlib.release(DS, MODULE, rows, meta, ck)


if __name__ == "__main__":
    main()
