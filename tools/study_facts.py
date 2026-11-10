"""Summarise each published dataset for the site's tags.

Reads the public download files in source/release/*.zip (never the corpus) and writes
source/study_facts.json: for every dataset, the distinct values of the columns that describe
how the study was run (setting, device, population, reports, feedback, sessions ...).
"""
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
REL = HERE / "source/release"
manifest = json.load(open(REL / "manifest.json"))

CATS = ["setting", "device", "population", "hand_used", "instr_report", "probe_kind", "trial_kind", "fb_visible",
        "fb_form", "target_event", "instr_strategy", "instr_disclosure", "medication_state", "id_type", "instr_goal"]
USE = CATS + ["person", "group", "age", "session", "prep_time_lo_ms", "fb_delay_ms", "mirror_axis_deg", "gain",
              "outcome_channels", "cue_elements", "n_cursors", "aim_report", "hand_report", "rt_ms"]

facts = {}
for ds, m in sorted(manifest.items(), key=lambda kv: kv[1]["name"]):
    name = m["name"]
    with zipfile.ZipFile(REL / m["file"]) as z:
        df = pd.read_csv(io.TextIOWrapper(z.open(f"{name}/{name}.csv"), encoding="utf-8"), usecols=USE, low_memory=False)
    f = {"catalog_study": m["catalog_study"], "rows": len(df)}
    for c in CATS:
        f[c] = sorted(str(v) for v in df[c].dropna().unique())
    for c in ["outcome_channels", "cue_elements"]:
        toks = set()
        for v in df[c].dropna().unique():
            toks.update(t.split(":")[0] for t in str(v).split("|") if t)
        f[c] = sorted(toks)
    per = df.drop_duplicates("person")
    ages = pd.to_numeric(per["age"], errors="coerce")
    f["age_known"] = int(ages.notna().sum())
    f["age_median"] = None if ages.isna().all() else float(ages.median())
    f["group_age_median"] = {}
    if per["group"].notna().any():
        for g, sub in per.groupby("group"):
            a = pd.to_numeric(sub["age"], errors="coerce")
            if a.notna().sum() >= 3:
                f["group_age_median"][str(g)] = float(a.median())
    f["sessions_max"] = int(pd.to_numeric(df["session"], errors="coerce").max())
    f["forced_prep"] = bool(df["prep_time_lo_ms"].notna().any())
    f["fb_delay"] = bool((pd.to_numeric(df["fb_delay_ms"], errors="coerce") > 0).any())
    f["mirror"] = bool(df["mirror_axis_deg"].notna().any())
    f["gain"] = bool(df["gain"].notna().any())
    f["multi_cursor"] = bool((pd.to_numeric(df["n_cursors"], errors="coerce") > 1).any())
    f["aim_reports"] = int(df["aim_report"].notna().sum())
    f["hand_reports"] = int(df["hand_report"].notna().sum())
    f["rt"] = bool(df["rt_ms"].notna().any())
    facts[name] = f
    print(name, len(df), f["setting"], f["device"], f["population"], f["age_median"], f["sessions_max"], flush=True)

json.dump(facts, open(HERE / "source/study_facts.json", "w"), indent=1)
print("wrote", len(facts), "datasets")
