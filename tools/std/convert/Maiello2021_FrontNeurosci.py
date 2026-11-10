"""Maiello, Schepko, Klein, Paulun & Fleming 2021 (Front Neurosci): judging which of two marked precision grips on a 3D polycube
object is better, by vision alone (imagined grasps), after watching videos of other people's grasps, or after performing both
grasps (Experiment 1: 21 participants, 16 objects; Experiment 2a: 25 participants, 6 objects; Experiment 2b: 25 participants,
6 objects, three sessions). One row is one judgement.

Deposit (Zenodo 4382477): a 5.2 GB Data_and_Analyses.zip, mostly ZED stereo videos (.svo, Experiment 2a), model penalty maps and
analysis code. Only the small per-participant MATLAB files (DataExp01/02a/02b/S###_*.mat, about 10 KB each), Results/Stats_Results.txt
and ReadMe.txt were read out of it with HTTP range requests; nothing from the deposit is executed. The grasping movements themselves
were filmed in Experiment 2a only; the videos are not converted.

Run from the site folder: python3 -I tools/std/convert/Maiello2021_FrontNeurosci.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Maiello2021_FrontNeurosci"
MODULE = "perception"
RAW = stdlib.raw_dir(DS)
OBJ_OF_COND_2 = [3, 4, 6, 9, 14, 16]  # Experiments 2a/2b use these six objects of Experiment 1 (Experiment_02a.m: CondNumsExp1)
DEVICE = "Chin rest, 3D-printed polycube objects (wood and brass) on a turntable, verbal or typed report"
STIM = "which of two marked precision grips on the object is better (green or blue thumb sticker)"


def rows_for(exp, files, sessions):
    rows = []
    people = []
    for f in files:
        m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["SubjectInfo"]
        sid = f"{exp}_S{int(Path(f).name.split('__S')[1][:3]):03d}"  # file number: SubjectInfo.ID repeats in Experiment 1 (file 6 carries ID 2)
        sex = {"M": "M", "F": "F"}.get(str(m.gender).strip().upper()[:1], "other")
        hand = {"R": "right", "L": "left"}.get(str(m.handedness).strip().upper()[:1], "")
        order = np.atleast_1d(m.CondOrder).astype(int)
        correct = str(m.CorrectResp).upper()
        go = [str(x) for x in np.atleast_1d(m.GraspOrder)] if hasattr(m, "GraspOrder") else None
        n = len(order)
        assert sorted(order) == list(range(1, n + 1)) and len(correct) == n
        people.append(dict(sid=sid, sex=sex, age=int(m.age), hand=hand))
        t = 0
        for si, (sname, field) in enumerate(sessions, start=1):
            resp = str(getattr(m, field)).upper()
            assert len(resp) == n and set(resp) <= {"G", "B"}, (sid, field, resp)
            for cond in order:
                t += 1
                obj = OBJ_OF_COND_2[cond - 1] if exp != "E1" else int(cond)
                first = ""
                if go is not None and (exp != "E1" or sname == "grasping"):
                    first = {"G": ", green grasp first", "B": ", blue grasp first"}.get(go[cond - 1][:1], "")
                r = resp[cond - 1]
                rows.append(dict(
                    dataset=DS, study=DS, experiment={"E1": "Experiment 1", "E2a": "Experiment 2a", "E2b": "Experiment 2b"}[exp], group="",
                    subject=sid, person=sid, subject_excluded_by_authors=False, age=int(m.age), sex=sex, handedness=hand,
                    population="healthy", setting="lab", device=DEVICE, session=si, block=sname, trial=t, phase="test", time_s="",
                    task="grasping and lifting", unit="trial", condition=sname + first, pert_type="none", pert_value=0, pert_unit="",
                    feedback="normal", excluded_by_authors=False,
                    stimulus=STIM, stimulus_value=obj, stimulus_unit="object number (Experiment 1 numbering, 1-16)", reference_value="",
                    response=r, response_label={"G": "green grasp judged better", "B": "blue grasp judged better"}[r],
                    response_value="", response_unit="", response_correct=(r == correct[cond - 1]), reaction_time_s="", bout=""))
    return rows, people


def pc(df, sess, sub=None):
    d = df[df.block == sess]
    if sub is not None:
        d = d[d.stimulus_value.isin(sub)]
    return d.groupby("subject").response_correct.mean() * 100


def main():
    ck = stdlib.Checks()
    rows = []
    allp = {}
    sess1 = [("vision", "ResponseSession1"), ("grasping", "ResponseSession2")]
    sess2b = [("vision", "ResponseSession1"), ("video", "ResponseSession2"), ("grasping", "ResponseSession3")]
    for exp, folder, sessions in (("E1", "Experiment_01__DataExp01", sess1), ("E2a", "Experiment_02a__DataExp02a", sess1), ("E2b", "Experiment_02b__DataExp02b", sess2b)):
        files = sorted(RAW.glob(f"{folder}__S*.mat"))
        r, p = rows_for(exp, files, sessions)
        rows += r
        allp[exp] = p
    df = pd.DataFrame(rows)
    n = {e: len(p) for e, p in allp.items()}
    ck.published("Experiment 1 participants (21)", n["E1"], 21, 0, "Methods, Participants")
    ck.published("Experiment 2a participants (25)", n["E2a"], 25, 0, "Methods, Participants")
    ck.published("Experiment 2b participants (25)", n["E2b"], 25, 0, "Methods, Participants")
    for e, (nf, nm, mage) in {"E1": (16, 5, 24), "E2a": (17, 8, 23), "E2b": (18, 7, 24)}.items():
        ck.published(f"{e} women ({nf})", sum(q["sex"] == "F" for q in allp[e]), nf, 0, "Methods, Participants")
        ck.published(f"{e} men ({nm})", sum(q["sex"] == "M" for q in allp[e]), nm, 0, "Methods, Participants")
        ck.published(f"{e} mean age ({mage})", np.mean([q["age"] for q in allp[e]]), mage, 0.5, "Methods, Participants (mean reported as a whole number)")
    ck.published("Experiment 1 youngest and oldest age (19, 32)", min(q["age"] for q in allp["E1"]) * 100 + max(q["age"] for q in allp["E1"]), 1932, 0, "Methods, Participants")
    ck.published("Experiment 2b youngest and oldest age (19, 36)", min(q["age"] for q in allp["E2b"]) * 100 + max(q["age"] for q in allp["E2b"]), 1936, 0, "Methods, Participants")
    d1, d2a, d2b = (df[df.experiment == e] for e in ("Experiment 1", "Experiment 2a", "Experiment 2b"))

    def t1(x, mu=50):
        r = stats.ttest_1samp(x, mu)
        return r.statistic

    def tp(a, b):
        return stats.ttest_rel(a, b).statistic

    v, g = pc(d1, "vision"), pc(d1, "grasping")
    ck.published("Exp 1 vision vs chance t(20) = 6.63", t1(v), 6.63, 0.01, "Results, Experiment 1")
    ck.published("Exp 1 grasping vs chance t(20) = 15.79", t1(g), 15.79, 0.01, "Results, Experiment 1")
    ck.published("Exp 1 grasping vs vision t(20) = 5.14", tp(g, v), 5.14, 0.01, "Results, Experiment 1")
    vs, gs = pc(d1, "vision", OBJ_OF_COND_2), pc(d1, "grasping", OBJ_OF_COND_2)
    ck.published("Exp 1 subset vision vs chance t(20) = 0.5", t1(vs), 0.5, 0.05, "Results, Experiment 2 (the six conditions of Experiment 1 used again)")
    ck.published("Exp 1 subset grasping vs chance t(20) = 10.25", t1(gs), 10.25, 0.01, "Results, Experiment 2")
    ck.published("Exp 1 subset grasping vs vision t(20) = 4.81", tp(gs, vs), 4.81, 0.01, "Results, Experiment 2")
    v, g = pc(d2a, "vision"), pc(d2a, "grasping")
    ck.published("Exp 2a vision vs chance t(24) = 1.88", t1(v), 1.88, 0.01, "Results, Experiment 2a")
    ck.published("Exp 2a grasping vs chance t(24) = 7.27", t1(g), 7.27, 0.01, "Results, Experiment 2a")
    ck.published("Exp 2a grasping vs vision t(24) = 3.51", tp(g, v), 3.51, 0.01, "Results, Experiment 2a")
    v, vd, g2 = pc(d2b, "vision"), pc(d2b, "video"), pc(d2b, "grasping")
    ck.published("Exp 2b vision vs chance t(24) = -1.19", t1(v), -1.19, 0.01, "Results, Experiment 2b")
    ck.published("Exp 2b video vs chance t(24) = 4.58", t1(vd), 4.58, 0.01, "Results, Experiment 2b")
    ck.published("Exp 2b grasping vs chance t(24) = 6.41", t1(g2), 6.41, 0.01, "Results, Experiment 2b")
    ck.published("Exp 2b video vs vision t(24) = 4.23", tp(vd, v), 4.23, 0.01, "Results, Experiment 2b")
    ck.published("Exp 2b grasping vs vision t(24) = 6.35", tp(g2, v), 6.35, 0.01, "Results, Experiment 2b")
    ck.published("Exp 2b grasping vs video t(24) = 0.92", tp(g2, vd), 0.92, 0.01, "Results, Experiment 2b")
    ck.published("Grasping session, Exp 2b vs Exp 2a, t(24) = 0.68 in magnitude", abs(tp(g2.values, pc(d2a, 'grasping').values)), 0.68, 0.01, "Results, Experiment 2b (participants matched to the Experiment 2a participant whose videos they saw)")
    ck.internal("rows = judgements: 21 x 32 + 25 x 12 + 25 x 18", len(df) == 21 * 32 + 25 * 12 + 25 * 18, str(len(df)))
    ck.internal("every response is G or B", set(df.response) == {"G", "B"})
    meta = dict(
        study=DS,
        paper=dict(authors="Maiello G, Schepko M, Klein LK, Paulun VC, Fleming RW", year=2021,
                   title="Humans Can Visually Judge Grasp Quality and Refine Their Judgments Through Visual and Haptic Feedback",
                   venue="Front Neurosci 14:591898", doi="10.3389/fnins.2020.591898"),
        lab="Fleming (Justus Liebig University Giessen)", data_url="https://doi.org/10.5281/zenodo.4382477",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Only the small per-participant MATLAB files of the three experiments were read out of the deposit's 5.2 GB ZIP (with HTTP range requests). The strings of responses per session were unpacked to one row per judgement, in the order "
                 "each participant saw the objects (the deposit's CondOrder). Letters were upper-cased (the deposit mixes cases), participant numbers became codes, and the file date stamps were dropped. "
                 "Not included: the ZED stereo videos of Experiment 2a (about 5 GB of the ZIP, filmed grasps), the model penalty maps and perturbed models, the analysis code, and the thumb and index length fields (every participant has the placeholder value 99)."),
        how=["One row is one judgement of which of two marked precision grips on one object was better. Experiment 1: 21 participants judged 16 objects twice, by vision alone (imagining both grasps) and after physically performing both grasps (sessions 1 and 2). "
             "Experiment 2a: 25 new participants judged 6 of those objects by vision and after grasping. Experiment 2b: 25 further participants judged the same 6 objects by vision, after watching videos of Experiment 2a participants' grasps, and after grasping (sessions 1, 2 and 3).",
             "stimulus_value is the object number in Experiment 1's numbering (Experiments 2a and 2b use objects 3, 4, 6, 9, 14 and 16). response is G (green grasp judged better) or B (blue); response_correct is true when the response matches the authors' CorrectResp, "
             "the grasp the authors' model rates as closer to optimal. The grasps themselves (and in Experiment 2a their video) are not part of the table; where the deposit records which grasp came first, condition says so (Experiment 1 records it only for the grasping session).",
             "Movement is not recorded in the table: the reported quantity is a two-alternative choice made after imagining, watching or performing a precision grip."],
        unit_note="judgement of grasp quality",
        differences=["In Experiment 1 two files (numbers 2 and 6) carry the same SubjectInfo.ID; participants are separated by file number, which gives the 21 participants of the paper (the deposit's own analysis also loops over files).",
                     "The paper (Methods, Experiment 1) says verbal reports; the deposit's scripts collected the letters G or B by an input box, the table has those letters.",
                     "Experiment 2b's grasping session (session 3) is its own participants' judgements, not a copy of Experiment 2a's."],
    )
    stdlib.release(DS, MODULE, df, meta, ck)


if __name__ == "__main__":
    main()
