"""Klein, Maiello, Paulun & Fleming 2020 (PLoS Comput Biol): precision-grip grasps of novel objects in three experiments (12, 14 and 14
participants). Deposit (Zenodo 3891663): a 36 GB Data_and_Analyses.zip, mostly model penalty maps and perturbed models. Only small human-data
files were read out of it with range requests: Experiments1_2/00_Data/CSV_Data/DataExp1.txt and DataExp2.txt (per grasp: object, angle, start
position, thumb and index contact points, and 600-sample trajectories) and Experiment3/00_Data/S0xx.mat (contact points per grasp).
The 600-sample trajectories are not converted: the deposit gives no sampling rate or time base for them."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Klein2020_PLoSComputBiol"
MODULE = "grasp_contacts"
PFX = "Data_and_Analyses__"
DEVICE12 = "Optotrak 3020 with infrared markers on the thumb and index fingernails"
DEVICE3 = "Optotrak Certus with infrared markers on the thumb and index fingernails"
CONTACT = ["thumb_contact_x_mm", "thumb_contact_y_mm", "thumb_contact_z_mm", "index_contact_x_mm", "index_contact_y_mm", "index_contact_z_mm"]
V2 = ["thumb_contact_v2_x_mm", "thumb_contact_v2_y_mm", "thumb_contact_v2_z_mm", "index_contact_v2_x_mm", "index_contact_v2_y_mm", "index_contact_v2_z_mm"]


def v(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def core(exp, sid, sex, age, hand, device, block, trial, phase, cond):
    return dict(dataset=DS, study=DS, experiment=exp, group="", subject=sid, person=sid, subject_excluded_by_authors=False,
                age=age, sex=sex, handedness=hand, population="healthy", setting="lab", device=device, session=1, block=block,
                trial=trial, phase=phase, time_s="", task="grasping and lifting", unit="trial", condition=cond, pert_type="none",
                pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False)


def main():
    ck = stdlib.Checks()
    raw = stdlib.raw_dir(DS)
    rows = []
    cnt = {}
    for e in (1, 2):
        df = pd.read_csv(raw / f"{PFX}Experiments1_2__00_Data__CSV_Data__DataExp{e}.txt", usecols=range(14))
        df.columns = ["sub", "trial", "main", "obj", "angle", "sx", "sy", "sz", "tx", "ty", "tz", "ix", "iy", "iz"]
        exp = f"Experiment {e}"
        for sub, g in df.groupby("sub", sort=True):
            g = g.sort_values("trial", kind="stable")
            sid = f"E{e}_S{int(sub):02d}"
            for r in g.itertuples(index=False):
                d = core(exp, sid, "", "", "right", DEVICE12, "main" if r.main == 1 else "practice", int(r.trial),
                         "test" if r.main == 1 else "practice", f"object {int(r.obj)}, angle {r.angle:g} deg")
                d.update(object_id=int(r.obj), object_angle_deg=float(r.angle), start_x_mm=float(r.sx), start_y_mm=float(r.sy), start_z_mm=float(r.sz),
                         thumb_contact_x_mm=float(r.tx), thumb_contact_y_mm=float(r.ty), thumb_contact_z_mm=float(r.tz),
                         index_contact_x_mm=float(r.ix), index_contact_y_mm=float(r.iy), index_contact_z_mm=float(r.iz))
                rows.append(d)
            cnt[(e, int(sub))] = (int((g.main == 1).sum()), int((g.main == 0).sum()))
    # Experiment 3
    sexes, ages, nerr, nmain, nprac = [], [], 0, 0, 0
    perm_ok, cond_ok = [], []
    files = sorted(raw.glob(f"{PFX}Experiment3__00_Data__S0*.mat"))
    for f in files:
        m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        si, T, P, E = m["SubjectInfo"], m["Test"], m["Practice"], m["Error"]
        sid = "E3_S" + f.name.split("__S0")[1][:2]
        sex = {"M": "M", "F": "F"}.get(str(si.gender).strip().upper()[:1], "other")
        hand = {"R": "right", "L": "left"}.get(str(si.handedness).strip().upper()[:1], "")
        age = int(si.age)
        sexes.append(sex); ages.append(age)
        reps = int(si.Reps)
        order = np.asarray(si.ShuffledTrials).astype(int)
        perm_ok.append(sorted(order) == list(range(1, len(order) + 1)) and len(order) == 28)
        conds_obj = np.asarray(si.Conds).astype(int)
        cond_ok.append(all(conds_obj[k] == k // reps + 1 for k in range(len(conds_obj))))
        nerr += 0 if (E is None or not hasattr(E, "Trial")) else int(np.size(E.Trial))
        tr = 0
        for q in np.asarray(P.ContactT_V01).reshape(-1, 3).tolist() and range(len(np.asarray(P.ContactT_V01).reshape(-1, 3))):
            tr += 1
            ct, ci = np.asarray(P.ContactT_V01).reshape(-1, 3)[q], np.asarray(P.ContactI_V01).reshape(-1, 3)[q]
            c2t, c2i = np.asarray(P.ContactT_V02).reshape(-1, 3)[q], np.asarray(P.ContactI_V02).reshape(-1, 3)[q]
            d = core("Experiment 3", sid, sex, age, hand, DEVICE3, "practice", tr, "practice", "practice grasp")
            d.update({k: v(x) for k, x in zip(CONTACT, list(ct) + list(ci))}); d.update({k: v(x) for k, x in zip(V2, list(c2t) + list(c2i))})
            rows.append(d); nprac += 1
        CT, CI = np.asarray(T.ContactT_V01), np.asarray(T.ContactI_V01)
        C2T, C2I = np.asarray(T.ContactT_V02), np.asarray(T.ContactI_V02)
        for k in np.argsort(order, kind="stable"):
            obj = int(conds_obj[k])
            d = core("Experiment 3", sid, sex, age, hand, DEVICE3, "main", int(order[k]) + len(np.asarray(P.ContactT_V01).reshape(-1, 3)), "test", f"object {obj}")
            d.update(object_id=obj)
            d.update({kk: v(x) for kk, x in zip(CONTACT, list(CT[k]) + list(CI[k]))}); d.update({kk: v(x) for kk, x in zip(V2, list(C2T[k]) + list(C2I[k]))})
            rows.append(d); nmain += 1
    n_files = len(files)
    ck.internal("Experiment 1: 12 participants with 80 main and 6 practice grasps", sum(1 for k in cnt if k[0] == 1) == 12 and all(v_ == (80, 6) for k, v_ in cnt.items() if k[0] == 1))
    ck.internal("Experiment 2: 14 participants with 84 main and 6 practice grasps", sum(1 for k in cnt if k[0] == 2) == 14 and all(v_ == (84, 6) for k, v_ in cnt.items() if k[0] == 2))
    ck.internal("Experiment 3: ShuffledTrials is a permutation of 1..28 and Conds group 7 repetitions per object", all(perm_ok) and all(cond_ok))
    ck.published("Experiment 1 participants (12)", sum(1 for k in cnt if k[0] == 1), 12, 0, "Methods, Experiment 1, Participants")
    ck.published("Experiment 2 participants (14)", sum(1 for k in cnt if k[0] == 2), 14, 0, "Methods, Experiment 2, Participants")
    ck.published("Experiment 3 participants (14)", n_files, 14, 0, "Methods, Experiment 3, Participants; Stats_Results.txt")
    ck.published("Experiment 1 main grasps per participant (80)", np.mean([v_[0] for k, v_ in cnt.items() if k[0] == 1]), 80, 0, "Methods, Experiment 1 (16 conditions x 5 repetitions)")
    ck.published("Experiment 2 main grasps per participant (84)", np.mean([v_[0] for k, v_ in cnt.items() if k[0] == 2]), 84, 0, "Methods, Experiment 2 (36 conditions x 7 repetitions)")
    ck.published("Experiment 3 main grasps per participant (28)", nmain / n_files, 28, 0, "Methods, Experiment 3 (4 objects x 7 repetitions)")
    ck.published("Experiment 3 practice grasps per participant (5)", nprac / n_files, 5, 0, "Methods, Experiment 3")
    ck.published("Experiment 3 males (5)", sum(s == "M" for s in sexes), 5, 0, "Results, Experiment 3 participants (Stats_Results.txt)")
    ck.published("Experiment 3 females (9)", sum(s == "F" for s in sexes), 9, 0, "Results, Experiment 3 participants (Stats_Results.txt)")
    ck.published("Experiment 3 mean age (25.1)", np.mean(ages), 25.1, 0.05, "Stats_Results.txt, Experiment 3")
    ck.published("Experiment 3 youngest and oldest ages (19, 58)", min(ages) + 100 * max(ages), 19 + 5800, 0, "Stats_Results.txt, Experiment 3")
    pct = 100 * nerr / (nerr + n_files * 28)
    ck.published("Experiment 3 percentage error trials (6.9)", pct, 6.9, 0.05, "Stats_Results.txt, Experiment 3 (error trials / all trials)")
    df = pd.DataFrame(rows)
    meta = dict(
        study=DS,
        paper=dict(authors="Klein LK, Maiello G, Paulun VC, Fleming RW", year=2020,
                   title="Predicting precision grip grasp locations on three-dimensional objects",
                   venue="PLoS Comput Biol 16(8):e1008081", doi="10.1371/journal.pcbi.1008081"),
        lab="Fleming (Justus Liebig University Giessen)", data_url="https://doi.org/10.5281/zenodo.3891663",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Only the small human-data files were read out of the deposit's 36 GB ZIP (with range requests): the CSV exports of Experiments 1 and 2 (first 14 columns: subject, trial, practice/main, object, angle, start position, "
                 "thumb and index contact points) and the per-participant MATLAB files of Experiment 3 (contact points V01 and V02 of the practice and test grasps). Columns were renamed, units added, and participant numbers became codes "
                 "(date-stamped file names and experiment dates dropped). Experiment 3 test grasps were put in presentation order (the deposit's ShuffledTrials). "
                 "Not included: the 600-sample finger trajectories of Experiments 1 and 2 and the trajectories of Experiment 3 (the deposit gives no sampling rate or time base for them), the four hand-recorded error-trial grasps per participant group of Experiment 3, "
                 "and the model penalty maps and perturbed models (about 35 GB)."),
        how=["One row is one grasp of one object with thumb and index finger by one participant, in the order of the deposit (Experiments 1 and 2: trial number 1-86 and 1-90, where the first 6 are practice; Experiment 3: 5 practice grasps then the 28 test grasps in presentation order).",
             "Contact points are the 3D positions of the thumb and index fingertips at first contact in the motion tracking system's coordinates, in mm (unit assumed from the distances in mm the paper reports). Experiments 1 and 2 give object number, "
             "the object's orientation angle in degrees and the hand's start position; the deposit's CSV export does not name the object numbers (the paper uses four shapes L, U, S, V at two orientations in Experiment 1 and 12 composite objects in Experiment 2).",
             "Experiment 3 has four 3D-printed objects (paper: swan, blob, cat, croissant; object_id 1-4 is the deposit's numbering, and the deposit does not say which number is which) and a second version of each contact point (V02), computed by the authors' code as a point on the object's reachable surface. "
             "Age and sex are only available for Experiment 3 (in the participants' files); the paper reports Experiments 1 and 2 as all right-handed."],
        unit_note="grasp (precision grip)",
        differences=["Participant IDs of the three experiments are coded separately (E1_, E2_, E3_ prefixes); the deposit does not say whether anyone took part in more than one experiment (the paper calls all participants naive).",
                     "In Experiments 1 and 2 the CSV has exactly the designed 80/84 test grasps plus 6 practice grasps per participant, although the paper reports 13.8% and 13.9% error trials: error trials were repeated and are not in the CSV. "
                     "In Experiment 3 the error trials were repeated too; their hand-recorded locations (Error struct) are not converted."],
    )
    stdlib.release(DS, MODULE, df, meta, ck)


if __name__ == "__main__":
    main()
