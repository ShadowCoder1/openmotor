"""van Polanen, Buckingham & Davare (2022), J Neurophysiol 128(2):290-301: 47 adults received continuous theta-burst TMS over aIPS, M1 or
a low-intensity sham over aIPS and then lifted 4 cubes (small/large x light/heavy) 81 times, giving a heaviness estimate after each lift
(size-weight illusion). OSF https://osf.io/u9vq4/. Converted: the authors' per-trial table (Processed_data.csv, 81 trials x 47
participants), their last practice trial (Processed_lastpractice.csv) and per-participant MEPs (Processed_MEP_data.csv, to participants.csv).
The raw force traces (3 ZIPs of ~300 MB) are not converted: the authors' per-trial force measures are.
Run on the PC: python3 -I tools/std/pc.py run vanPolanen2022_JNeurophysiol"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "vanPolanen2022_JNeurophysiol"
B = stdlib.raw_dir(DS)
GROUP = {1: "aIPS cTBS", 2: "Sham cTBS", 3: "M1 cTBS"}
INTENS = {1: 80, 2: 40, 3: 80}
SIZE = {1: ("small", 5.0), 2: ("large", 10.0), 3: ("medium", 7.5)}
WEIGHT = {1: ("light", 190), 2: ("heavy", 400), 3: ("light", 190)}


def main():
    d = pd.read_csv(B / "Processed_data.csv")
    lp = pd.read_csv(B / "Processed_lastpractice.csv")
    mep = pd.read_csv(B / "Processed_MEP_data.csv")
    d = d.rename(columns={"TMS group": "g"}); lp = lp.rename(columns={"TMS group": "g"}); mep = mep.rename(columns={"TMSgroup": "g"})
    ck = stdlib.Checks()
    rows = []
    for (g, pp), t in d.groupby(["g", "pp"], sort=True):
        subj = f"G{g}_P{pp}"
        p = lp[(lp.g == g) & (lp.pp == pp)]
        assert len(p) == 1
        seq = [(p.iloc[0], True)] + [(r, False) for _, r in t.sort_values("tr").iterrows()]
        prev_w = float("nan")
        for i, (r, prac) in enumerate(seq):
            sz, cm = SIZE[int(r["size"])]
            wc, wg = WEIGHT[int(r["weight"])]
            removed = (not prac) and int(r["removed"]) == 1
            f = lambda c: float("nan") if (removed or pd.isna(r[c])) else float(r[c])  # noqa: E731
            row = dict(dataset=DS, study="van Polanen 2022 J Neurophysiol", experiment="", group=GROUP[g], subject=subj, person=subj,
                       subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy", setting="lab",
                       device="Manipulandum with two ATI Nano17 force sensors on magnetically attached cubes; cTBS with Brainsight neuronavigation",
                       session=1, block="practice" if prac else "test", trial=i + 1, phase="practice" if prac else "test", time_s="",
                       task="grasping and lifting", unit="trial", condition=f"{sz} {wc}",
                       pert_type="cTBS", pert_value=INTENS[g], pert_unit="% of active motor threshold",
                       feedback="normal", excluded_by_authors=removed,
                       series=float("nan"), perceived_heaviness=float("nan"), perceived_heaviness_z=float("nan"), lift=int(r["tr"]), weight_g=wg, prev_weight_g=prev_w, object_size_cm=cm,
                       dur_load_phase_s=f("LPD"), first_peak_load_force_rate_n_s=f("LFR1st"), first_peak_grip_force_rate_n_s=f("GFR1st"))
            if not prac:
                row["perceived_heaviness"] = float(r["Perception"]); row["perceived_heaviness_z"] = float(r["Perception (z-scored)"])
            prev_w = wg
            rows.append(row)
    part = pd.DataFrame(dict(person=[f"G{g}_P{p}" for g, p in zip(mep.g, mep.pp)], tms_group=[GROUP[g] for g in mep.g],
                             pre_mep_uv=mep.preMEP, post_mep_uv=mep.postMEP))
    pcols = [("tms_group", "text", "", "cTBS group."),
             ("pre_mep_uv", "number", "microvolts", "Mean peak-to-peak MEP amplitude (first dorsal interosseus) before cTBS, as the authors averaged it; empty for the participant the authors removed."),
             ("post_mep_uv", "number", "microvolts", "Mean MEP amplitude after cTBS. The authors judged the MEP recordings possibly unreliable and did not analyse them.")]
    df = pd.DataFrame(rows)
    test = df[df.phase == "test"]
    ck.published("participants (47)", df.subject.nunique(), 47, 0, "Methods, Participants")
    for g, n in [(1, 16), (3, 16), (2, 15)]:
        ck.published(f"{GROUP[g]} group size ({n})", df[df.group == GROUP[g]].subject.nunique(), n, 0, "Methods, Participants")
    ck.published("test trials per participant (81 = 4 objects x 20 + 1)", test.groupby("subject").size().mean(), 81, 0, "Methods, Grasp and lift task")
    ck.published("trials removed from perception and force analysis (15)", int(test.excluded_by_authors.sum()), 15, 0, "Methods, Force and perceptual analysis")
    has_force = test.groupby("subject").first_peak_load_force_rate_n_s.apply(lambda s: s.notna().any())
    ck.published("participants with force data (47 - 2 = 45; F(1,44))", int(has_force.sum()), 45, 0, "Methods, Participants; Results, Force Parameters")
    for g in (1, 2, 3):
        ck.published(f"{GROUP[g]}: participants with force data (15)", int(has_force[[s.startswith(f'G{g}_') for s in has_force.index]].sum()), 15, 0, "Figure 2 legend (n = 15 for all groups)")
    t12 = test[test.trial.isin([2, 3])]
    ok = sum(1 for s, t in t12.groupby("subject") if list(t.sort_values("trial").condition) == ["small light", "large light"])
    ck.published("participants whose first two lifts were small-light then large-light (47)", ok, 47, 0, "Methods, Grasp and lift task")
    cnt = test.groupby(["subject", "condition"]).size().unstack()
    ck.internal("each object lifted 20 or 21 times by every participant", bool(cnt.min().min() >= 20 and cnt.max().max() <= 21), str(cnt.describe().loc[["min", "max"]].to_dict()))
    ck.internal("size-weight illusion direction: small objects rated heavier (z) than large of same weight",
                bool(test[test.condition == "small light"].perceived_heaviness_z.mean() > test[test.condition == "large light"].perceived_heaviness_z.mean()))
    ck.internal("heavy lifted with longer loading phase than light", bool(test[test.condition.str.endswith("heavy")].dur_load_phase_s.mean() > test[test.condition.str.endswith("light")].dur_load_phase_s.mean()))
    ck.internal("row count = 47 x 82", len(df) == 47 * 82)
    meta = dict(study=DS, paper=dict(authors="van Polanen V, Buckingham G, Davare M", year=2022,
                title="The effects of TMS over the anterior intraparietal area on anticipatory fingertip force scaling and the size-weight illusion",
                venue="J Neurophysiol 128(2):290-301", doi="10.1152/jn.00265.2021"),
        lab="Davare (KU Leuven / Brunel), Buckingham (Exeter)", data_url="https://osf.io/u9vq4/",
        license="none stated", license_url="",
        changes=("The authors' Processed_data.csv (81 test trials), Processed_lastpractice.csv and Processed_MEP_data.csv were converted; codes were replaced by meanings "
                 "(group 1/2/3 = aIPS/Sham/M1 cTBS, size 1/2/3 = small 5 cm/large 10 cm/medium 7.5 cm, weight 1/2/3 = light 190 g/heavy 400 g/practice 190 g), the last practice trial was placed before the test trials, "
                 "and the deposit's zeros in force columns for the 15 removed trials (placeholders) were left empty. The raw force traces (three ZIPs of about 300 MB) and the perception text files are not included."),
        how=["One row is one lift of a cube with thumb and index finger after cTBS: trial 1 is the last practice lift (medium cube, no heaviness estimate), trials 2-82 are the 81 test lifts in the authors' order.",
             "Forces are the authors' own: first peak of the load force rate and of the grip force rate after grip force onset and loading-phase duration in seconds. The deposit lists the rates as [N]; N/s is inferred because they are first derivatives of force.",
             "perceived_heaviness is the magnitude estimate on the participant's own scale, perceived_heaviness_z its z-score as the authors computed. pert_type cTBS with the stimulation intensity in % of active motor threshold; participants.csv holds the MEP means (which the authors did not analyse)."],
        unit_note="lift (practice or test trial)",
        differences=["Rows flagged excluded_by_authors are the 15 trials the authors removed from the perception and force analysis; their perception value is kept as given, forces are empty.",
                     "Two participants (one aIPS, one M1) have no force values because of technical errors, as in the paper; their perception values are present and their rows are not flagged.",
                     "Ages and sex per participant are not in the deposit (paper: 27 of 47 female, mean age 22 years)."])
    stdlib.release(DS, "grasp_lift", rows, meta, ck, participants=part, participant_columns=pcols)


if __name__ == "__main__":
    main()
