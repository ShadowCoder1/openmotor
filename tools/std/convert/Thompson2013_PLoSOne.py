"""Thompson, Blair, Chen & Henrey (2013), PLoS ONE: SkillCraft1 master table, one row per StarCraft 2 game (one per player).

Run from the site folder: python3 -I tools/std/convert/Thompson2013_PLoSOne.py
Raw file (untrusted data, read only): raw_extra/Thompson2013_PLoSOne/skillcraft.zip from
https://archive.ics.uci.edu/dataset/272 (doi 10.24432/C5161N).
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Thompson2013_PLoSOne"
checks = stdlib.Checks()
z = zipfile.ZipFile(stdlib.raw_dir(ID) / "skillcraft.zip")
raw = pd.read_csv(z.open("SkillCraft1_Dataset.csv"), na_values=["?"])
n = len(raw)
LEAGUE = {1: "Bronze", 2: "Silver", 3: "Gold", 4: "Platinum", 5: "Diamond", 6: "Master", 7: "GrandMaster", 8: "Professional"}

checks.internal("3395 rows as on the UCI page", n == 3395)
checks.internal("GameID unique, in increasing order (the file order is kept)", raw.GameID.is_unique and raw.GameID.is_monotonic_increasing)
checks.internal("league index 1-8 only", set(raw.LeagueIndex) <= set(range(1, 9)))
checks.internal("missing values only in age and the two hours columns (professional players)",
                raw.drop(columns=["Age", "HoursPerWeek", "TotalHours"]).notna().all().all()
                and set(raw.LeagueIndex[raw.Age.isna()]) == {8})

t = pd.DataFrame({
    "dataset": ID, "study": "Thompson et al. 2013 PLoS ONE", "experiment": "SkillCraft1", "group": raw.LeagueIndex.map(LEAGUE),
    "subject": [f"P{g}" for g in raw.GameID], "subject_excluded_by_authors": False,
    "age": raw.Age, "sex": "", "handedness": "", "population": "healthy", "setting": "online",
    "device": "keyboard and mouse (StarCraft 2 replay)", "session": 1, "block": "", "trial": 1, "phase": "other", "time_s": "",
    "task": "sports and games", "unit": "trial", "condition": "", "pert_type": "none", "pert_value": 0, "pert_unit": "",
    "feedback": "normal", "excluded_by_authors": False,
    "game": "StarCraft 2", "attempt": raw.GameID,
    "apm": raw.APM, "select_by_hotkeys": raw.SelectByHotkeys, "assign_to_hotkeys": raw.AssignToHotkeys,
    "unique_hotkeys": raw.UniqueHotkeys, "minimap_attacks": raw.MinimapAttacks, "minimap_right_clicks": raw.MinimapRightClicks,
    "pacs_per_timestamp": raw.NumberOfPACs, "gap_between_pacs": raw.GapBetweenPACs, "action_latency": raw.ActionLatency,
    "actions_in_pac": raw.ActionsInPAC, "map_explored": raw.TotalMapExplored, "workers_made": raw.WorkersMade,
    "unique_units_made": raw.UniqueUnitsMade, "complex_units_made": raw.ComplexUnitsMade,
    "complex_abilities_used": raw.ComplexAbilitiesUsed, "league_index": raw.LeagueIndex,
    "hours_per_week": raw.HoursPerWeek, "total_hours": raw.TotalHours,
})
for c in ("age", "hours_per_week", "total_hours"):
    t[c] = t[c].astype(object).where(t[c].notna(), None)

S = "Data Collection / Results (Participants)"
cnt = raw.LeagueIndex.value_counts()
for i, (nm, rep) in enumerate([("Bronze", 167), ("Silver", 347), ("Gold", 553), ("Platinum", 811), ("Diamond", 806),
                               ("Masters", 621)], start=1):
    checks.published(f"players in the {nm} league", cnt[i], rep, 0, "Analysis, 'The sample size by league'")
checks.published("professional games", cnt[8], 55, 0, "Analysis, 'The sample size by league' and Data Collection")
checks.published("players excluding the 35 GrandMasters the paper dropped", n - cnt[7], 3360, 0, "Abstract; Data Collection")
checks.published("age mean (years)", raw.Age.mean(), 21.6, 0.05, "Data Collection (mean 21.6)")
checks.published("age SD (years)", raw.Age.std(), 4.2, 0.05, "Data Collection (SD 4.2)")
checks.published("age median (years)", raw.Age.median(), 21, 0, "Data Collection (median 21)")
checks.published("age minimum and maximum: minimum", raw.Age.min(), 16, 0, "Data Collection (16-44)")
checks.published("age maximum", raw.Age.max(), 44, 0, "Data Collection (16-44)")
SUP = "Supporting Information, Figs S1-S15"
for col, nm, m, s in [("APM", "actions per minute", 117.05, 51.95), ("ActionsInPAC", "actions per PAC", 5.27, 1.49),
                      ("UniqueHotkeys", "unique hotkeys", 4.36, 2.36), ("TotalMapExplored", "map explored", 22.13, 7.43),
                      ("UniqueUnitsMade", "unique units made", 6.53, 1.86)]:
    checks.published(f"{nm}: mean", raw[col].mean(), m, 0.005, SUP)
    checks.published(f"{nm}: SD", raw[col].std(), s, 0.005, SUP)
checks.published("action latency mean in seconds (timestamps / 88.5; paper 719.94 ms, mean only to 0.01 s here)",
                 raw.ActionLatency.mean() / 88.5, 0.71994, 0.0005, SUP + " (S1)")

meta = dict(
    study="Thompson et al. 2013",
    paper=dict(authors="Thompson JJ, Blair MR, Chen L, Henrey AJ", year=2013,
               title="Video Game Telemetry as a Critical Tool in the Study of Complex Skill Learning",
               venue="PLoS ONE 8(9):e75129", doi="10.1371/journal.pone.0075129"),
    lab="Blair (Simon Fraser University)", data_url="https://doi.org/10.24432/C5161N",
    license="CC BY 4.0 (UCI Machine Learning Repository page; the OpenML mirror lists a different licence)",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Converted to the OpenMotor game format with the authors' column names mapped to snake_case; GameID kept as attempt. No values changed.",
    how=["One row is one StarCraft 2 game played by one player (the paper says each participant contributed a single game). "
         "league_index and group give the player's skill league on the game's ladder (1 Bronze to 8 Professional).",
         "The telemetry columns are the authors' per-game measures. Counts marked per timestamp are per replay timestamp "
         "(about 88.5 per second at the faster game speed); gap_between_pacs and action_latency are in timestamps too "
         "(divide by 88.5 for seconds). apm is actions per minute.",
         "Age, hours_per_week and total_hours are the players' self-reports; they are empty for the professional players.",
         "trial is 1 for every row (one game per player); there is no date, so session and time are not given."],
    unit_note="game (one per player)",
    differences=["The UCI file keeps 35 GrandMaster games (league 7) that the paper left out of its analysis (3,360 players there, 3,395 here).",
                 "The paper's supplementary latency and gap values are in milliseconds and seconds and its rates are per minute; "
                 "this file holds the UCI versions in replay timestamps or per timestamp, which were not converted.",
                 "The paper's exclusion counts sum to a different total than it states, and the UCI page notes that some players may have "
                 "submitted more than one game; the file has no player identifier, so every row is treated as a different person."],
)
stdlib.release(ID, "game", t.to_dict("records"), meta, checks)
