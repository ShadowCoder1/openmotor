"""Nagamine (2023), Proc 20th International Congress of Phonetic Sciences (ICPhS 2023), Prague: "Dynamic tongue movements in L1
Japanese and L2 English liquids". Ultrasound tongue imaging of 17 L1 Japanese and 12 L1 North American English speakers reading
believe, bereave and (Japanese speakers) biribiri; the tongue surface was tracked by DeepLabCut in AAA as 11 points, at 11
equidistant time points over the vowel-liquid-vowel interval. One row is one token.

Raw files, raw_extra/Nagamine2023_ICPhS/ (OSF 29tac, no licence set on the node):
  ICPhS_df.csv        the authors' merge of the AAA exports: 3036 lines, one per token and time point (speaker code, recording time stamp,
                      prompt, interval label, 11 x/y pairs along the tongue from root to tip, language); 33 lines are empty, one token
                      (11 lines) has no coordinates
  ICPhS_df_xy_pca.csv the authors' analysis table (2981 lines = 271 tokens): equals ICPhS_df.csv without the empty lines, the token with
                      no coordinates and the one token the script drops as mislabelled (rec time 2022/11/29 17:48:39); used as a check
  ICPhS_supplement.Rmd the authors' script (read as text); wiki: results replicate after the 3 August 2023 correction
Not needed: ICPhS_df_long_data_viz.csv (a long copy), the per-speaker spline text files (already merged), the HTML.
Run from the site folder: python3 -I tools/std/convert/Nagamine2023_ICPhS.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Nagamine2023_ICPhS"
SRC = "Nagamine 2023, ICPhS"
DEVICE = ("Telemed MicrUs ultrasound (64-element probe, 20 mm radius, 2-4 MHz, 80 mm depth, about 80 frames per second) in a "
          "stabilisation headset, Articulate Assistant Advanced 220.4.1; audio through a USBPre2 interface at 44.1 kHz")
MIS = "2022/11/29 17:48:39"
V = [f"V{i}" for i in range(5, 27)]
R = stdlib.raw_dir(ID)
ck = stdlib.Checks()
d = pd.read_csv(R / "ICPhS_df.csv")
x = pd.read_csv(R / "ICPhS_df_xy_pca.csv")


def f(v):
    return "" if pd.isna(v) else format(float(v), ".8g")


empty = d.V1.isna()
m = d[~empty].copy()
ck.internal("33 empty lines (no speaker, word or coordinates) are left out; 3003 lines = 273 tokens x 11 time points",
            int(empty.sum()) == 33 and len(m) == 3003 and m.groupby(["V1", "V2"], sort=False).size().eq(11).all())
blk = (m[["V1", "V2"]] != m[["V1", "V2"]].shift()).any(axis=1).cumsum()
ck.internal("each token is one contiguous block of 11 lines", blk.nunique() == m.groupby(["V1", "V2"]).ngroups)
rows, ctr = [], {}
for _, g in m.groupby(blk, sort=True):
    a = g.iloc[0]
    sp = a.V1
    ctr[sp] = ctr.get(sp, 0) + 1
    xs = g[V[0::2]].to_numpy(float)
    ys = g[V[1::2]].to_numpy(float)
    nanfr = bool(np.isnan(xs).any())
    reasons = []
    if a.V2 == MIS:
        reasons.append("token the authors' script drops as not labelled properly")
    if nanfr:
        reasons.append("no tongue coordinates (dropped by the script's removal of missing values)")
    word = "biribiri" if str(a.V3) == "18" else a.V3
    jp = a.language == "Japanese"
    rows.append(dict(
        dataset=ID, study=SRC, experiment="", group="L1 Japanese" if jp else "L1 English", subject=sp, person=sp,
        subject_excluded_by_authors=False, age="", sex="", handedness="", time_s="", population="healthy", setting="lab", device=DEVICE, session=1, block="", trial=ctr[sp],
        phase="test", task="speech", unit="utterance", condition=word, pert_type="none", pert_value=0, pert_unit="", feedback="normal",
        excluded_by_authors=bool(reasons), exclusion_reason="; ".join(reasons), stimulus=word, target_consonant=str(a.V4)[1],
        vowel_context="i_i", n_contour_points=11, n_frames=11, frame_times_pct=";".join(str(10 * i) for i in range(11)),
        contour_frames_x=";".join(f(v) for v in xs.ravel()), contour_frames_y=";".join(f(v) for v in ys.ravel())))
out = pd.DataFrame(rows)
ck.internal("273 tokens; 271 not excluded; excluded flags are 2 tokens", len(out) == 273 and int((~out.excluded_by_authors).sum()) == 271)
# compare with the authors' analysis table
kept = m[(m.V2 != MIS)].dropna(subset=V)
ck.internal("analysis table ICPhS_df_xy_pca.csv equals the non-excluded lines (2981 lines, same values)",
            len(kept) == len(x) and np.allclose(kept[V].to_numpy(float), x[[f"{c}_{i}" for i in range(1, 12) for c in "XY"]].to_numpy(float)))
P = "Nagamine 2023"
ppl = out.drop_duplicates("person")
ck.published("L1 Japanese speakers", (ppl.group == "L1 Japanese").sum(), 17, 0, f"{P}, Abstract and 2.1 Participants")
ck.published("L1 North American English speakers", (ppl.group == "L1 English").sum(), 12, 0, f"{P}, Abstract and 2.1 Participants")
ck.published("tokens for analysis", int((~out.excluded_by_authors).sum()), 271, 0, f"{P}, 2.3 Data collection (271 tokens)")
ck.published("time points per token", 11, 11, 0, f"{P}, 2.4 Data analysis (11 equidistant time points)")
X = kept[V].to_numpy(float)
X = X - X.mean(0)
ev = np.linalg.eigvalsh(X.T @ X / len(X))[::-1]
sh = ev / ev.sum() * 100
ck.published("variance explained by PC1 (%), PCA of the 22 coordinates of the converted rows", sh[0], 58.03, 0.01, f"{P}, 3.1 (PC1: 58.03%)")
ck.published("variance explained by PC2 (%)", sh[1], 22.65, 0.01, f"{P}, 3.1 (PC2: 22.65%)")
ck.published("variance explained by PC3 (%)", sh[2], 7.65, 0.02, f"{P}, 3.1 (PC3: 7.65%)")
ck.published("variance explained by PC1-3 (%)", sh[:3].sum(), 88.33, 0.02, f"{P}, 3.1 (88.33%)")
for c in ck.items:
    if not c["ok"]:
        print("FAILED:", c)
meta = dict(
    paper=dict(authors="Nagamine T", year=2023, title="Dynamic tongue movements in L1 Japanese and L2 English liquids",
               venue="Proc 20th International Congress of Phonetic Sciences (ICPhS 2023), Prague", doi=""),
    lab="Nagamine (Lancaster University)", data_url="https://osf.io/29tac/", license="none stated", license_url="",
    changes=("Took the authors' merged AAA table ICPhS_df.csv and turned its 3003 non-empty lines (273 tokens x 11 time points) into 273 rows, "
             "one per token, with the 11 contours as two lists (x and y, 11 time points x 11 points). The 33 empty lines were left out. "
             "The two tokens the authors' script removes (one dropped as mislabelled, one with no coordinates) are kept and flagged. "
             "The recording time stamps were used only to tell tokens apart and are not copied; speaker codes are replaced by new codes; "
             "the prompt the table shows as 18 (biribiri, which AAA could not write in Japanese script) is called biribiri, as in the authors' script."),
    how=["One row is one token: one speaker reading believe (interval ili), bereave (iri) or, for Japanese speakers, the Japanese "
         "word biribiri (interval idi, the Japanese tap). target_consonant is the middle letter of the authors' AAA interval label "
         "(l, r, d) and vowel_context i_i because both flanking vowels are /i/. group is the speaker's first language.",
         "contour_frames_x and contour_frames_y hold the 11 tongue points (from the tongue root to the tip, in the authors' order) at "
         "11 equidistant time points from the start (0 %) to the end (100 %) of the vowel-liquid-vowel interval: the first 11 numbers are "
         "the contour at 0 %, the next 11 at 10 %, and so on. The authors do not state the unit or direction of x and y.",
         "trial follows the order of the authors' table within a speaker; their age and sex are not given per speaker (the paper reports "
         "17 Japanese speakers, 8 women and 9 men, mean age 19.76, and 12 English speakers, 10 women and 2 men, mean age 29.08)."],
    unit_note="token (one word said once, with its tongue contour at 11 time points in two list columns)",
    differences=["Participant counts, 271 tokens and the PCA shares (PC1 58.03 %, PC2 22.65 %, PC3 7.66 % here against 7.65 %) are reproduced "
                 "by a PCA (centred, covariance matrix) of the 22 raw coordinates of the 271 tokens, as in the authors' script; the paper's text says "
                 "the data were z-scored per speaker first, which gives different shares, so the script (no z-scoring) matches the paper.",
                 "The deposit has no licence on the OSF node; the paper itself is open access (CC BY 4.0, UCL Discovery)."],
)
if not ck.passed():
    sys.exit("checks failed")
stdlib.release(ID, "tongue_ultrasound", out.to_dict("records"), meta, ck)
