"""Dokovova, Sugden, Cartney, Schaeffler & Cleland (2023), J Speech Lang Hear Res 66(7):2164-2183: "Tongue shape complexity in
children with and without speech sound disorders". Ultrasound tongue imaging of 29 typically developing (TD) children and 30
children with speech sound disorders (SSD) saying the syllables /aCa/ (children with SSD ten times each, TD children once each);
the authors fitted one tongue contour (42 points) in the frame of maximal lingual gesture of each token and computed the number
of inflections (NINFL) and the modified curvature index (MCI) of it. One row is one token.

Raw files, raw_extra/Dokovova2023_JSLHR/ (OSF 9cwfg, node licence CC-By Attribution 4.0 International):
  new_OSF_data.csv     the table of the revised paper (3061 rows, 60 children, 11 consonants): per token the contour (X1..X42, Y1..Y42),
                       NINFL as output by the software (ninfl), MCI, the authors' listening judgement (Error), the repetition, the time of
                       the frame in its recording, and the child's code, age, sex, scores. The paper analyses six of the 11 phonemes.
  OSF_code_revision1.R the authors' analysis script for that table (read as text): it drops five phonemes, one child, NINFL above 5
                       (after adding 1) and MCI above 6
  preprint_OSF_data.csv, preprint_OSF_code.R  the earlier table and script of the preprint (2783 rows); not converted
Not converted (not needed or not documented): the Fourier-style columns real_/imag_/mod_ (not described in the paper), the listener-rating
columns (named with the raters' initials), recording dates and times, client codes, the recruitment column, the Diagnosis column (blank or
inconsistent for 7 children; the paper's Appendix B is used instead), and the deposit's PDF of the paper.
Appendices A and B of the paper (demographics of the 29 TD and 30 SSD children) are copied below to check the table against.
Run from the site folder: python3 -I tools/std/convert/Dokovova2023_JSLHR.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Dokovova2023_JSLHR"
SRC = "Dokovova et al. 2023, J Speech Lang Hear Res"
SIX = ["a", "l", "sh", "r", "s", "t"]  # phonemes analysed in the paper (a = the vowel of /apa/, the reference)
DEVICE = {
    "TD": ("Sonix RP ultrasound machine (about 120 frames per second, 134 degree field of view), microconvex probe (10 mm radius) in a "
           "metal headset under the chin, with synchronized audio"),
    "SSD": ("Ultrasound Micro machine run from a laptop with SonoSpeech software (about 100 frames per second, 119 to 162 degree "
            "field of view), microconvex probe (10 mm radius) in an UltraFit plastic headset under the chin, with synchronized audio"),
}

# Appendix A of the paper: typically developing children, sex, age (years;months) and BPVS standard score
APPX_A = """02TD M 11;10 94|03TD M 11;9 115|04TD F 10;5 94|06TD M 9;11 113|07TD F 9;10 93|08TD F 9;4 86|09TD M 8;8 129|10TD F 6;9 115
11TD F 11;3 102|12TD M 8;1 119|13TD M 6;8 120|14TD F 11;7 94|15TD M 12;4 104|16TD M 7;11 117|17TD F 12;10 121|18TD M 10;8 116
19TD F 7;2 98|20TD M 12;6 105|21TD M 11;1 108|22TD F 5;8 92|23TD M 7;2 99|24TD F 12;2 85|25TD F 8;7 85|26TD M 10;0 104
27TD F 7;11 117|28TD M 11;5 76|29TD F 7;3 107|30TD F 7;11 84|31TD F 10;5 72"""
# Appendix B: children with speech sound disorders, sex, age, speech disorder reported by the speech and language therapist
APPX_B = """LN01|M|8;2|Phonological disorder
LN02|F|6;3|Articulation disorder
LN03|M|9;2|Articulation disorder
LN04|M|10;9|Articulation disorder
LN05|M|5;9|Articulation disorder
LN06|F|9;0|no report
LN07|M|6;3|Motor speech disorder
LN08|F|6;3|Phonological disorder
LN09|M|5;11|Inconsistent phonological disorder or CAS
LN10|M|6;4|Inconsistent phonological disorder
LN11|F|5;2|Diagnostic process ongoing
LN12|M|8;2|Motor speech disorder
GR01|M|9;1|Articulation disorder
GR02|M|9;2|Inconsistent speech sound disorder, query CAS
GR03|M|8;11|Residual speech sound errors
GR04|M|6;2|Articulation disorder
GR05|M|7;5|Articulation disorder
GR06|M|7;5|Articulation and phonological disorder with possible coordination difficulties
GR07|M|7;11|Articulation disorder
GR08|M|6;0|Possible CAS
GR09|M|12;11|Phonological disorder
GR11|M|8;4|Articulation disorder
GR12|F|11;6|Motor speech disorder
GR13|F|11;2|Articulation disorder
GR14|M|10;10|Articulation disorder
GR15|F|7;4|Articulation disorder
GR17|M|8;5|Articulation disorder
GR18|M|7;10|Articulation disorder
GR19|F|7;8|Articulation disorder
GR20|M|7;8|Articulation disorder"""


def ym(s):
    y, m = s.split(";")
    return int(y) * 12 + int(m)


appx_a = {c: dict(sex=sx, months=ym(a), bpvs=int(b)) for c, sx, a, b in (t.split() for t in APPX_A.replace("\n", "|").split("|"))}
appx_b = {c: dict(sex=sx, months=ym(a), disorder=dis) for c, sx, a, dis in (t.split("|") for t in APPX_B.splitlines())}

R = stdlib.raw_dir(ID)
ck = stdlib.Checks()
d = pd.read_csv(R / "new_OSF_data.csv", low_memory=False)
XC = [f"X{i}" for i in range(1, 43)]
YC = [f"Y{i}" for i in range(1, 43)]
assert all(c in d.columns for c in XC + YC)
ck.internal("3061 rows, 60 children, 42 contour points per row, no missing coordinate",
            len(d) == 3061 and d["name.x"].nunique() == 60 and not d[XC + YC].isna().any().any())
ck.internal("child, age, sex, BPVS and DEAP values are constant within a child",
            bool((d.groupby("name.x")[["age_y", "age_m", "sex", "BPVS_SS", "DEAP_PCC"]].nunique(dropna=False) <= 1).all().all()))
ck.internal("x falls and y is not constant from point 1 to point 42 in the table (mean x 116 at point 1, 44 at point 42)",
            bool(d.X1.mean() > d.X42.mean() + 50))

# ---- the authors' exclusions, as in OSF_code_revision1.R ---------------------------------------------------------------------
d["ninfl1"] = d.ninfl + 1
is_td = d["name.x"].str.endswith("TD")
drop_phoneme = ~d.phoneme.isin(SIX)
drop_child = d["name.x"] == "05TD"  # DEAP percentage consonants correct 91 (script: "DEAP is <100")
drop_ninfl = (d.ninfl1 > 5) & ~drop_phoneme & ~drop_child
drop_mci = (d.MCI > 6) & ~drop_phoneme & ~drop_child & ~(d.ninfl1 > 5)  # the script filters NINFL first, then MCI
reasons = []
for i in range(len(d)):
    r = []
    if drop_phoneme.iloc[i]:
        r.append("consonant not analysed in the paper (the authors' script drops j, th, p, k and w)")
    if drop_child.iloc[i]:
        r.append("child excluded: DEAP percentage consonants correct 91, below 100")
    if (d.ninfl1.iloc[i] > 5) and not drop_phoneme.iloc[i] and not drop_child.iloc[i]:
        r.append("NINFL above 5")
    if drop_mci.iloc[i]:
        r.append("MCI above 6")
    reasons.append("; ".join(r))
d["reason"] = reasons

# ---- rows ------------------------------------------------------------------------------------------------------------------
def f(v):
    return format(float(v), ".8g")


order = {}
rows = []
cols = {c: i for i, c in enumerate(d.columns)}
arr = d.to_numpy(dtype=object)
for k in range(len(d)):
    g = arr[k]
    code = g[cols["name.x"]]
    grp = "TD" if code.endswith("TD") else "SSD"
    order[code] = order.get(code, 0) + 1
    ph = g[cols["phoneme"]]
    xs = ";".join(f(g[cols[c]]) for c in XC)
    ys = ";".join(f(g[cols[c]]) for c in YC)
    rows.append(dict(
        dataset=ID, study=SRC, experiment="", group=grp, subject=code, person=code,
        subject_excluded_by_authors=bool(code == "05TD"), age=(int(g[cols["age_y"]]) * 12 + int(g[cols["age_m"]])) / 12,
        sex=g[cols["sex"]], handedness="", population="healthy" if grp == "TD" else "speech sound disorder",
        setting="lab" if grp == "TD" else "clinic", device=DEVICE[grp], session=1, block="", trial=order[code], phase="test",
        time_s="", task="speech", unit="utterance", condition=ph, pert_type="none", pert_value=0, pert_unit="", feedback="normal",
        excluded_by_authors=bool(d.reason.iloc[k] != ""),
        stimulus="apa (first vowel)" if ph == "a" else f"aCa, C = {ph}", target_consonant=ph,
        vowel_context="a (vowel of /apa/)" if ph == "a" else "a_a", repetition=int(g[cols["rep"]]), n_contour_points=42,
        contour_x=xs, contour_y=ys, frame_time_s=float(g[cols["Time.of.sample.in.recording"]]), ninfl_aaa=int(g[cols["ninfl"]]),
        mci=float(g[cols["MCI"]]), perceptual_error_code=str(g[cols["Error"]]).strip(), exclusion_reason=d.reason.iloc[k],
    ))
out = pd.DataFrame(rows)

# participants.csv (one row per child)
sp = d.drop_duplicates("name.x").set_index("name.x")
people = []
for code, r in sp.iterrows():
    people.append(dict(person=code, bpvs_standard_score=(None if pd.isna(r.BPVS_SS) else int(r.BPVS_SS)),
                       deap_pcc=(None if pd.isna(r.DEAP_PCC) else float(r.DEAP_PCC)),
                       reported_disorder=(appx_b[code]["disorder"] if code in appx_b else "")))
PCOLS = [("bpvs_standard_score", "integer", "", "British Picture Vocabulary Scale standard score (typically developing children only; the authors' BPVS_SS)."),
         ("deap_pcc", "number", "percent", "Percentage consonants correct on the Diagnostic Evaluation of Articulation and Phonology screen (typically developing children only; the authors' DEAP_PCC)."),
         ("reported_disorder", "text", "",
          "Speech disorder reported by the child's speech and language therapist (children with speech sound disorders only), as listed in Appendix B of the paper; 'no report' is the paper's dash. The authors' data table leaves this blank for 6 children and shows a dash for another.")]

# ---- checks against the paper ----------------------------------------------------------------------------------------------
P = "Dokovova et al. 2023"
ppl = out.drop_duplicates("person")
td = ppl[(ppl.group == "TD") & ~ppl.subject_excluded_by_authors]
ssd = ppl[ppl.group == "SSD"]
ck.published("typically developing children analysed", len(td), 29, 0, f"{P}, Method, Participants (29 typically developing participants)")
ck.published("typically developing children recorded (30 consented, one excluded)", (ppl.group == "TD").sum(), 30, 0, f"{P}, Method, Participants (Thirty children ... consented)")
ck.published("typically developing girls", (td.sex == "F").sum(), 15, 0, f"{P}, Method, Participants (15 girls, 14 boys)")
ck.published("typically developing boys", (td.sex == "M").sum(), 14, 0, f"{P}, Method, Participants (15 girls, 14 boys)")
tdm = td.age * 12
ck.published("TD youngest age (months; 5;8)", tdm.min(), ym("5;8"), 0.01, f"{P}, Method, Participants (aged between 5;8 and 12;10)")
ck.published("TD oldest age (months; 12;10)", tdm.max(), ym("12;10"), 0.01, f"{P}, Method, Participants (aged between 5;8 and 12;10)")
ck.published("TD mean age (months; 9;8 = 116)", tdm.mean(), ym("9;8"), 1.0, f"{P}, Method, Participants (M = 9;8; printed to the month)")
ck.published("TD SD of age (years; printed as 2)", tdm.std() / 12, 2, 0.1, f"{P}, Method, Participants (SD = 2)")
ck.published("children with speech sound disorders analysed", len(ssd), 30, 0, f"{P}, Method, Participants (a total of 30 children)")
ck.published("SSD boys", (ssd.sex == "M").sum(), 22, 0, f"{P}, Method, Participants (22 boys and eight girls)")
ck.published("SSD girls", (ssd.sex == "F").sum(), 8, 0, f"{P}, Method, Participants (22 boys and eight girls)")
sm = ssd.age * 12
ck.published("SSD oldest age (months; 12;11)", sm.max(), ym("12;11"), 0.01, f"{P}, Method, Participants (5;0-12;11)")
ck.published("SSD mean age (months; 8;2 = 98)", sm.mean(), ym("8;2"), 1.0, f"{P}, Method, Participants (M = 8;2; printed to the month)")
ck.published("SSD SD of age (years; printed as 2)", sm.std() / 12, 2, 0.1, f"{P}, Method, Participants (SD = 2)")
dis = pd.Series({c: v["disorder"] for c, v in appx_b.items()})
for key, n, label in (("Articulation disorder", 16, "articulation disorder"), ("Motor speech disorder", 3, "motor speech disorder"),
                      ("Phonological disorder", 3, "phonological disorder"), ("Residual speech sound errors", 1, "residual speech sound errors"),
                      ("Inconsistent phonological disorder", 1, "inconsistent phonological disorder"), ("Diagnostic process ongoing", 1, "diagnostic process ongoing"),
                      ("no report", 1, "no report")):
    ck.published(f"SSD children reported with {label} (Appendix B)", (dis == key).sum(), n, 0, f"{P}, Method, Participants (diagnoses of the 30 children)")
ck.published("SSD children with other, inconclusive diagnoses (Appendix B)", len(dis) - sum((dis == k).sum() for k in (
    "Articulation disorder", "Motor speech disorder", "Phonological disorder", "Residual speech sound errors", "Inconsistent phonological disorder",
    "Diagnostic process ongoing", "no report")), 4, 0, f"{P}, Method, Participants (the remaining children had inconclusive diagnoses)")
ck.published("TD children with BPVS standard scores 72 and 76 (outside normal limits)", int(td.person.isin(sp.index[sp.BPVS_SS.isin([72, 76])]).sum()), 2, 0,
             f"{P}, Method, Participants (two exceptions, 72 and 76)")
ck.published("TD child with DEAP percentage consonants correct 91", (sp.DEAP_PCC == 91).sum(), 1, 0, f"{P}, Method, Participants (a 91 percentage consonants correct score)")
ok_a = sum(1 for c, v in appx_a.items() if sp.loc[c, "sex"] == v["sex"] and 12 * sp.loc[c, "age_y"] + sp.loc[c, "age_m"] == v["months"] and sp.loc[c, "BPVS_SS"] == v["bpvs"])
ck.published("TD children whose sex, age and BPVS score equal Appendix A (of 29)", ok_a, 29, 0, f"{P}, Appendix A")
ok_b = sum(1 for c, v in appx_b.items() if sp.loc[c, "sex"] == v["sex"] and 12 * sp.loc[c, "age_y"] + sp.loc[c, "age_m"] == v["months"])
ck.published("SSD children whose sex and age equal Appendix B (of 30)", ok_b, 30, 0, f"{P}, Appendix B")
six = out[out.target_consonant.isin(SIX)]
ck.published("tokens of the six analysed phonemes, 29 TD and 30 SSD children (splines fitted)", len(six[six.subject != "05TD"]), 1694, 0,
             f"{P}, Method, Spline fitting (1,694 tokens)")
ck.published("tokens with NINFL above 5 removed", int(drop_ninfl.sum()), 74, 0, f"{P}, Method, Extraction of NINFL and MCI (n = 74)")
ck.published("tokens with MCI above 6 removed (after the NINFL filter)", int(drop_mci.sum()), 39, 0, f"{P}, Method, Extraction of NINFL and MCI (n = 39)")
ck.internal("1581 analysed tokens of the six phonemes remain after the authors' exclusions (1694 - 74 - 39)",
            int((~out.excluded_by_authors & out.target_consonant.isin(SIX)).sum()) == 1581)
ck.internal("every token left in the table belongs to one of the six analysed phonemes",
            bool(out[~out.excluded_by_authors].target_consonant.isin(SIX).all()))
ck.internal("each child's rows keep the authors' file order (trial counts from 1)", bool(out.groupby("subject").trial.apply(lambda s: (s.values == np.arange(1, len(s) + 1)).all()).all()))
months_bad = [c for c in sp.index if sp.loc[c, "months"] != 12 * sp.loc[c, "age_y"] + sp.loc[c, "age_m"]]
ck.internal("the table's months column disagrees with its own age_y and age_m for 3 children (all SSD)",
            len(months_bad) == 3 and all(not c.endswith("TD") for c in months_bad), f"{len(months_bad)} children")
for c in ck.items:
    if not c["ok"]:
        print("FAILED:", c)

meta = dict(
    paper=dict(authors="Dokovova M, Sugden E, Cartney G, Schaeffler S, Cleland J", year=2023,
               title="Tongue shape complexity in children with and without speech sound disorders",
               venue="J Speech Lang Hear Res 66(7):2164-2183", doi="10.1044/2023_JSLHR-22-00472"),
    lab="Cleland (University of Strathclyde)", data_url="https://osf.io/9cwfg/",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Took the authors' revised-paper table (new_OSF_data.csv), kept all 3061 rows in its order (one per tongue contour) and "
             "wrote the 42 contour points of each row as an x list and a y list in two columns. Kept the authors' NINFL (as output by "
             "the software, before their +1), MCI, listening judgement, repetition and frame time. Marked as the authors' exclusions the "
             "five consonants their script drops (278 tokens), the one child they dropped, the tokens with NINFL above 5 and with MCI above 6. "
             "Left out: the Fourier-style columns real_, imag_ and mod_ (not described in the paper), the listener-rating columns, "
             "recording dates and times, the clients' file codes, the recruitment column, the diagnosis column (blank or inconsistent for 7 "
             "children; the paper's Appendix B is used in participants.csv instead), and the earlier preprint table. Age in years is the "
             "authors' age in whole years and months divided by 12. Child codes are replaced by new codes."),
    how=["One row is one tongue contour: one syllable /aCa/ said by a child (children with speech sound disorders ten times per "
         "consonant, typically developing children once) while the ultrasound probe imaged the tongue from under the chin, with the "
         "tongue surface traced in the frame of the maximal lingual gesture of the consonant (for 'a' rows, the midpoint of the first "
         "vowel of /apa/, the reference). target_consonant is the authors' label of the phoneme (a is the vowel; sh and r stand for the "
         "consonants /ʃ/ and /ɹ/; th, j, w, p and k are five further consonants that the paper does not analyse). The contour is 42 "
         "points; the authors do not state the direction or unit of x and y (x falls with the point number in this table).",
         "ninfl_aaa is the number of inflections that the Articulate Assistant Advanced software outputs (0 = one curve), so the paper's "
         "NINFL is this plus 1 and values of NINFL above 5 were dropped; mci is the modified curvature index (dropped above 6). "
         "perceptual_error_code is the authors' label of their own listening judgement (V alone = correct in their script). "
         "frame_time_s is the time of the frame within its recording, not within the session.",
         "trial counts a child's rows in the order of the authors' table, which is grouped by consonant, so it is not the order of speaking "
         "(the recording times the authors give are minute-level and do not match their repetition numbers). Children with speech sound "
         "disorders were recorded in community clinics, typically developing children in a university laboratory."],
    unit_note="tongue contour (one syllable /aCa/ token, traced in one ultrasound frame, with its NINFL and MCI)",
    differences=[
        "The table's months column (the age used in the authors' models) disagrees with its own age_y and age_m columns, and with the ages in "
        "Appendix B of the paper, for 3 of the 30 children with speech sound disorders; age here comes from age_y and age_m, which match "
        "Appendix B for all 59 children. The paper's mean age for these children (8;2) is closer to the months column (8;1.7) than to "
        "Appendix B (8;1.2); both are within the month the checks allow.",
        "The paper gives the BPVS standard scores of the typically developing children as mean 103, SD 14; the 29 scores of Appendix A, "
        "which equal the table, give mean 102.2 and SD 14.7.",
        "The table's Diagnosis column is blank for 6 children and shows a dash for one whom Appendix B lists with an articulation disorder; "
        "participants.csv takes the diagnosis from Appendix B, which gives the paper's counts (16 articulation disorder ...). The table's "
        "recruited column also puts two children whose codes begin with LN under GR; it is not used.",
        "The paper gives the age range of the children with speech sound disorders as 5;0 to 12;11; the youngest child in the table and in Appendix B is 5;2.",
        "The paper says typically developing children produced each consonant once; the table has a second repetition (rep 2) for some of their tokens.",
        "The paper's mixed-model estimates (Table 1) were not recomputed.",
    ],
)
if not ck.passed():
    sys.exit("checks failed")
rec = stdlib.release(ID, "tongue_ultrasound", out.to_dict("records"), meta, ck, participants=people, participant_columns=PCOLS)
