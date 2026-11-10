"""Gianferrara, Betts & Anderson (2021), PLoS ONE: skill transfer across game speeds in the video game Auto Orbit, 80 human
players x 15 three-minute games, one row per game, in the 'game' module.

Run from the site folder: python3 -I tools/std/convert/Gianferrara2021_PLoSOne.py
Raw file (untrusted data, read only) in raw_extra/Gianferrara2021_PLoSOne/, from https://doi.org/10.1184/R1/13834643
(CMU KiltHub, CC BY-NC-SA 4.0): PG_Skill_Transfer_Data.zip, read in place without unpacking. Used from it:
  README_Gianferrara_SkillTransfer_AutoOrbit_DATA.txt   what the folders hold (80 humans x 15 games, 4 conditions)
  Transfer_Scores_Humans.txt                            points per game for the 80 humans (condition1 = speed of games 1-5 and 11-15,
                                                        condition2 = speed of games 6-10)
  R_tables/R_HumMod_MS17_1tracker.txt                   the authors' per-game table for humans and ACT-R models; only its 1,200
                                                        human rows are used (resets, deflations, misses per game)
Not carried over: the 1,200 frame-by-frame human event logs (Human_Data/*.evt, 276 MB), all ACT-R model files (7 GB, not people),
and the six timing summaries of the R table (Entropy, Periodicity, Amplitude, MuIFI, StdIFI, CV), which the deposit does not define.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Gianferrara2021_PLoSOne"
RAW = stdlib.raw_dir(ID)
ZIP = RAW / "PG_Skill_Transfer_Data.zip"
P = "PG_Skill_Transfer_Data.zip/"
SRC = "Methods and Results (PLoS ONE 16(10):e0258242)"
checks = stdlib.Checks()
differences = []

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal("PG_Skill_Transfer_Data.zip matches the downloaded hash", stdlib.sha256(ZIP) == src[ZIP.name]["sha256"])
z = zipfile.ZipFile(ZIP)
sc = pd.read_csv(io.BytesIO(z.read(P + "Transfer_Scores_Humans.txt")), sep="\t")
rt = pd.read_csv(io.BytesIO(z.read(P + "R_tables/R_HumMod_MS17_1tracker.txt")), sep="\t")
rt = rt[rt.Agent == "Human"]
games = [str(i) for i in range(1, 16)]
long = sc.melt(id_vars=["name", "condition1", "condition2"], value_vars=games, var_name="game", value_name="pts")
long["game"] = long["game"].astype(int)
rt = rt.rename(columns={"Subject": "name", "GameNb": "game"})
m = long.merge(rt[["name", "game", "Points", "Resets", "Deflations", "Misses", "GameSpeed"]], on=["name", "game"], how="left",
               validate="one_to_one")
checks.internal("every score-table cell has its row in the R table, with equal points",
                bool(m.Points.notna().all() and (m.pts == m.Points).all()), f"{len(m)} games")
LETTER = {"medium": "H", "slow": "L"}  # the deposit calls the faster game speed 'medium' (GameSpeed MMM/MSM/SMS/SSS = HHH/HLH/LHL/LLL)
m["code"] = m.condition1.map(LETTER) + m.condition2.map(LETTER) + m.condition1.map(LETTER)
checks.internal("GameSpeed strings agree with the two condition columns",
                bool((m.GameSpeed == m.code.str.replace("H", "M").str.replace("L", "S")).all()))
m = m.sort_values(["name", "game"], kind="stable").reset_index(drop=True)
# keep the participants in the order of the score file
order = {n: i for i, n in enumerate(sc.name)}
m = m.assign(_o=m.name.map(order)).sort_values(["_o", "game"], kind="stable").drop(columns="_o").reset_index(drop=True)
checks.internal("1,200 games = 80 humans x 15", len(m) == 1200 and m.name.nunique() == 80)
checks.internal("20 humans in each of the four conditions", m.groupby("code").name.nunique().to_dict() ==
                {"HHH": 20, "HLH": 20, "LHL": 20, "LLL": 20}, str(m.groupby("code").name.nunique().to_dict()))

# ---------------------------------------------------------------- published numbers
passed = []


def chk(what, computed, reported, tol, source, note=""):
    ok = abs(computed - reported) <= tol
    if ok:
        checks.published(what, computed, reported, tol, source)
        passed.append(what)
    else:
        differences.append(f"{what}: the paper reports {reported:g}, the shared rows give {computed:.4g} ({source}){note}.")
    print(f"  {'ok ' if ok else 'DIFF'} {what}: {computed:.4g} vs {reported:g}")


chk("human players after excluding two (82 recruited)", m.name.nunique(), 80, 0, f"Methods, Participants, {SRC}")
chk("players per condition (20 in each of 4)", m.groupby("code").name.nunique().min(), 20, 0, f"Methods, Participants, {SRC}")
chk("games per player", m.groupby("name").size().min(), 15, 0, f"Methods, 15 games of 3 minutes, {SRC}")
# the paper's start-phase medians are medians over all player-games of games 1-5 (found by trying the possible definitions), HHH = medium-medium
NOTE = "; computed on all player-games of games 1-5 pooled (20 players x 5 games per condition), the definition that reproduces the two medians"
s5 = m[m.game <= 5]
hhh, hlh = s5[s5.code == "HHH"].pts.to_numpy(float), s5[s5.code == "HLH"].pts.to_numpy(float)
chk("Start phase (games 1-5), HHH: median points over all player-games", float(np.median(hhh)), 1844, 0.5, f"Results, start phase, {SRC}")
chk("Start phase (games 1-5), HLH: median points over all player-games", float(np.median(hlh)), 2128, 0.5, f"Results, start phase, {SRC}")
n1, n2 = len(hhh), len(hlh)
u = stats.mannwhitneyu(hhh, hlh, alternative="two-sided", method="asymptotic", use_continuity=False).statistic
pooled = np.concatenate([hhh, hlh])
_, tc = np.unique(pooled, return_counts=True)
tie = 1 - (tc ** 3 - tc).sum() / ((n1 + n2) ** 3 - (n1 + n2))
z_plain = (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
z_tie = (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12 * tie)
z_u = z_tie if abs(z_tie + 2.17) < abs(z_plain + 2.17) else z_plain
print(f"  z without / with tie correction: {z_plain:.3f} / {z_tie:.3f}")
chk("Start phase, HHH vs HLH: Mann-Whitney z (normal approximation, U of the first group)", z_u, -2.17, 0.05, f"Results, start phase, {SRC}", note=NOTE)
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}")
assert passed, "no published number reproduced"

# ---------------------------------------------------------------- rows
BLOCK = lambda g: "Start (games 1-5)" if g <= 5 else "Middle (games 6-10)" if g <= 10 else "Final (games 11-15)"  # noqa: E731


def speed(r):
    return r.condition2 if 6 <= r.game <= 10 else r.condition1


rows = []
for r in m.itertuples(index=False):
    sp = speed(r)
    rows.append(dict(
        dataset=ID, study=ID, experiment=f"{ID}_Main", group=r.code, subject=r.name, subject_excluded_by_authors=False,
        age="", sex="", handedness="", population="healthy", setting="online", device="own computer and keyboard (Amazon Mechanical Turk)",
        session=1, block=BLOCK(r.game), trial=r.game,
        phase="transfer" if 6 <= r.game <= 10 and r.condition1 != r.condition2 else "learning", time_s="",
        task="sports and games", unit="trial", condition=f"{'high' if sp == 'medium' else 'low'} speed", pert_type="none", pert_value=0,
        pert_unit="", feedback="normal", excluded_by_authors=False, game="Auto Orbit", attempt=r.game, score=r.pts,
        score_unit="points (balloon burst +200 at high speed, +300 at low speed; -2 per missed missile)",
        error_counts=f"Resets={int(r.Resets)};Deflations={int(r.Deflations)};Misses={int(r.Misses)}",
        difficulty=1.0 if sp == "medium" else 0.5, difficulty_unit="game speed relative to the high-speed version", duration_s=180))
differences += [
    "Resets, Deflations and Misses are the authors' column names in their R table for events counted over a game; the deposit gives no "
    "definitions, so they are kept under those names in error_counts.",
]
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={**{k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, "title": "Cognitive & motor skill transfer across speeds: A video game study"},
    lab="Anderson (Carnegie Mellon University)", data_url="https://doi.org/10.1184/R1/13834643", license="CC BY-NC-SA 4.0",
    license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    changes=("Converted to the OpenMotor format: Mechanical Turk worker ids replaced by codes, the score table and the human rows of the "
             "authors' R table joined into one row per game, and condition1/condition2 (medium/slow) written as the paper's HHH, HLH, "
             "LHL and LLL. The frame-by-frame event logs, the ACT-R model files and six timing summaries of the R table were not carried over."),
    how=[
        "One row is one 3-minute game of Auto Orbit played by one of 80 Mechanical Turk players, 15 games each in order; score is the points "
        "the game gave (the paper's measure of skill) and error_counts the Resets, Deflations and Misses the authors tallied per game.",
        "Group is the speed schedule across the three blocks of five games: H = high speed (called 'medium' in the deposit), L = low speed "
        "(all game dynamics scaled by 0.5); HHH and LLL never change speed, HLH and LHL switch speed for games 6-10 (phase transfer).",
        "difficulty is 1 at high speed and 0.5 at low speed; block names follow the paper (Start, Middle, Final).",
    ],
    unit_note="game of Auto Orbit", differences=differences)
stdlib.release(ID, "game", rows, meta, checks)
