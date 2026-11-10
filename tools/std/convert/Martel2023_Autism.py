"""Martel et al. (2023), Autism: reach-grasp-and-displace of a light or heavy bottle with known or unknown weight in children with
autism (ASD), developmental coordination disorder (DCD) and typically developing (TD) children, 14 per group, Optotrak.
One row per trial of the authors' file data0920.txt (the last 8 trials per weight and knowledge cell, as analysed), module
reach_displace_weight.

Run from the site folder: python3 -I tools/std/convert/Martel2023_Autism.py
Raw (untrusted, read only): raw_extra/Martel2023_Autism/data0920.txt from https://osf.io/m56pa/ (tab separated, decimal comma).
Stats.R / Graphs.R were read as text only.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Martel2023_Autism"
RAW = stdlib.raw_dir(ID)
ck = stdlib.Checks()
d = pd.read_csv(RAW / "data0920.txt", sep="\t", decimal=",", encoding="latin1")
d["_o"] = range(len(d))

# ------------------------------------------------------------ published checks (paper = Participants, Table 1, Statistical analysis)
pp = d.drop_duplicates("subjects").set_index("subjects")
for g in ("ASD", "DCD", "TD"):
    ck.published(f"children in the {g} group", (pp.group == g).sum(), 14, 0, "Methods, Participants (3 groups of 14)")
ck.published("children in total", len(pp), 42, 0, "Methods, Participants (3 x 14); F(2,39) in the age comparison")
ck.published("girls in the three groups together", (pp.sex == "F").sum(), 12, 0, "Methods, Participants (2 + 6 + 4 girls)")
ck.published("girls in the DCD group", ((pp.group == "DCD") & (pp.sex == "F")).sum(), 6, 0, "Methods, Participants (DCD: 6 girls)")
FA = stats.f_oneway(*[pp.age[pp.group == g] for g in ("ASD", "DCD", "TD")])
for g, m, s in (("ASD", 10.0, 17 / 12), ("DCD", 10 + 7 / 12, 16 / 12), ("TD", 10 + 5 / 12, 18 / 12)):
    ck.published(f"mean age of {g} (years)", pp.age[pp.group == g].mean(), m, 0.5 / 12,
                 "Methods, Participants (mean age, printed to the month)")
    ck.published(f"SD of age of {g} (years)", pp.age[pp.group == g].std(), s, 0.5 / 12,
                 "Methods, Participants (SD, printed to the month)")
ck.published("M-ABC total, DCD mean", pp.MABC[pp.group == "DCD"].mean(), 22.8, 0.05, "Methods, Participants (DCD 22.8 +- 6.3)")
ck.published("M-ABC total, DCD SD", pp.MABC[pp.group == "DCD"].std(), 6.3, 0.05, "Methods, Participants (DCD 22.8 +- 6.3)")
ck.published("M-ABC total, ASD mean", pp.MABC[pp.group == "ASD"].mean(), 18.5, 0.05, "Methods, Participants (ASD 18.5 +- 8.9)")
ck.published("M-ABC total, ASD SD", pp.MABC[pp.group == "ASD"].std(), 8.9, 0.05, "Methods, Participants (ASD 18.5 +- 8.9)")
ck.published("trials analysed per child (last 8 of 15 per weight x knowledge cell, 4 cells)", d.groupby("subjects").size().iloc[0], 32, 0,
             "Statistical analysis (only the last eight trials per weight and condition)")
ck.published("children with a trial count of 32", (d.groupby("subjects").size() == 32).sum(), 42, 0, "Statistical analysis")
ck.published("cells per child with 8 trials", (d.groupby(["subjects", "condition"]).size() == 8).sum(), 168, 0,
             "Statistical analysis (8 trials x 4 cells x 42 children)")
ck.internal("total rows 1344 = 42 x 32", len(d) == 1344)
ck.internal("MT1 + MT2 = MTTOT in every row where all three exist",
            bool(((d.MT1 + d.MT2 - d.MTTOT).dropna().abs() <= 1).all()))
ck.internal("condition codes match weight x knowledge", bool(
    (d.condition == (d.weight.map({"light": 1, "heavy": 2}) + d.know.map({"known": 0, "unknown": 2}))).all()))
ck.internal("one ORDER value per child", bool((d.groupby("subjects").ORDER.nunique() == 1).all()))
ck.internal("participant-level columns are constant within child", bool(
    (d.groupby("subjects")[["age", "sex", "group", "MABC", "dexterity", "ball", "balance"]].nunique() == 1).all().all()))

# ------------------------------------------------------------ rows
POP = {"ASD": "autism spectrum disorder", "DCD": "developmental coordination disorder", "TD": "typically developing"}
MAP = dict(reach_acc_latency_ms="TPACC", reach_acc_amplitude_mm_s2="ACC", reach_vel_latency_ms="TPV1", reach_vel_amplitude_mm_s="VEL1",
           reach_dec_latency_ms="TPD1", reach_dec_amplitude_mm_s2="DEC1", grip_aperture_latency_ms="TPMGA", grip_aperture_max_mm="MGA",
           displace_acc_latency_ms="TPA3", displace_acc_amplitude_mm_s2="ACC3", displace_vel_latency_ms="TPV2",
           displace_vel_amplitude_mm_s="VEL2", displace_dec_latency_ms="TPD2", displace_dec_amplitude_mm_s2="DEC2",
           lift_height_mm="Z", movement_time_reach_ms="MT1", movement_time_displace_ms="MT2", movement_time_total_ms="MTTOT")
rows = []
n = {}
for r in d.itertuples(index=False):
    n[r.subjects] = n.get(r.subjects, 0) + 1
    row = dict(dataset=ID, study=ID, experiment="reach-to-displace", group=r.group, subject=r.subjects, person=r.subjects,
               subject_excluded_by_authors=False, age=r.age, sex={"G": "M", "F": "F"}[r.sex], handedness="",
               population=POP[r.group], setting="lab", device="Optotrak 3020 (300 Hz), light or heavy bottle",
               session=1, block=f"{r.know} {r.weight}", trial=n[r.subjects], phase="test", time_s="",
               task="grasping and lifting", unit="trial", condition=f"{r.weight} {r.know}",
               pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
               object_weight=r.weight, weight_knowledge=r.know, known_first_weight=r.ORDER.split()[0].lower(),
               condition_code=r.condition, session_trial=r.trial)
    for k, v in MAP.items():
        row[k] = getattr(r, v)
    rows.append(row)
ck.internal("32 rows for each of 42 children, row order of the deposit kept", len(rows) == 1344 and all(v == 32 for v in n.values()))

pcols = [("dexterity", "number", "points", "M-ABC manual dexterity subtest score (higher = worse)."),
         ("ball_skills", "number", "points", "M-ABC ball skills subtest score."),
         ("balance", "number", "points", "M-ABC balance subtest score."),
         ("mabc_total", "number", "points", "M-ABC total impairment score (sum of the three subtests)."),
         ("vci", "number", "index", "WISC verbal comprehension index; empty when no IQ test."),
         ("pri", "number", "index", "WISC perceptual reasoning index."),
         ("wmi", "number", "index", "WISC working memory index."),
         ("psi", "number", "index", "WISC processing speed index."),
         ("iq_total", "number", "", "Total IQ where only the total was available (some TD children).")]
parts = [dict(person=s, dexterity=g.dexterity, ball_skills=g.ball, balance=g.balance, mabc_total=g.MABC, vci=g.VCI, pri=g.PRI,
              wmi=g.WMI, psi=g.PSI, iq_total=g.IQ) for s, g in pp.iterrows()]
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Roy / Martel (Lyon Neuroscience Research Center, ImpAct)", data_url="https://osf.io/m56pa/",
    license="none stated", license_url="",
    changes="converted to the OpenMotor format; column names spelled out, decimal commas read as decimals, child codes replaced, "
            "boys/girls coded M/F; the group-level M-ABC and IQ scores moved to participants.csv; the R scripts are not included.",
    unit_note="trial",
    how=["One row is one trial of a child reaching for a bottle, grasping it and displacing it to a marked spot with the dominant hand "
         "(light 50 g or heavy 500 g; weight known in blocks of one weight, or unknown with the two bottles mixed). The deposit holds "
         "the last 8 trials of each of the 4 weight x knowledge cells per child, the trials the authors analysed (32 per child, 42 "
         "children: 14 ASD, 14 DCD, 14 TD). The rows keep the deposit's order; blank cells are values the authors could not extract.",
         "Columns are the authors' kinematic peaks of the wrist (latency and amplitude of maximum acceleration, velocity and "
         "deceleration) in the Reach and Displace phases, maximum finger aperture, lift height and movement times. The deposit has "
         "no data dictionary: the names (TPACC, ACC, TPV1, VEL1, TPD1, DEC1, TPMGA, MGA, TPA3, ACC3, TPV2, VEL2, TPD2, DEC2, Z, MT1, MT2, "
         "MTTOT) were matched to the parameter list in the paper's Methods; units were inferred (latencies in ms as in the Figure 1 "
         "legend; amplitudes in mm/s and mm/s^2 because the paper's m/s and m/s^2 and values like 700 for peak velocity and 4500 for "
         "peak acceleration agree only in millimetres); MT1, MT2 and MTTOT are taken as Reach, Displace and total duration because "
         "MT1 + MT2 = MTTOT in every row.",
         "Sex is coded G or F in the deposit; G is read as the French garcon (boy) and mapped to M, which fits the paper's 12 girls "
         "in total. Handedness is empty (the paper gives left-handers per group but not per child). Data were collected at 300 Hz, "
         "second-order Butterworth 10 Hz, with the authors' semi-automatic extraction."],
    differences=[f"The paper reports the age comparison across groups as F(2,39) = .624, p = .541; the deposit's ages give F = {FA.statistic:.3f}, p = {FA.pvalue:.3f}, so that test is not used as a check (the group means and SDs of age do reproduce).",
                 "The paper says 2 girls in the ASD group and 4 in the TD group; the deposit's sex column has 3 and 3 (the DCD group, 6, and "
                 "the total, 12, agree). The deposit is kept as shared.",
                 "The deposit's license is not stated (the OSF project shows none); the paper's published version is in Autism 28(2), 2024 "
                 "(online 2023). The data are shared with credit and removed on request.",
                 "The paper's text was read from the author's accepted manuscript (University of Padua repository)."],
)
stdlib.release(ID, "reach_displace_weight", rows, meta, ck, participants=parts, participant_columns=pcols)
for c in ck.items:
    if c["kind"] == "published":
        print(f"  P {c['what'][:70]} {c['computed']:.4f} vs {c['reported']}")
