"""Bishop & Kwak (2026), Front Psychol: pairs of adults drummed a regular beat (75 bpm) together on MIDI drum pads while ignoring a
'noisy' metronome (long-range, short-range or white-noise timing; consonant or dissonant pitch), with or without visual contact.
One row per drum strike.

Run from the site folder: python3 -I tools/std/convert/Bishop2026_FrontPsychol.py
Raw files (untrusted data, read only) in raw_extra/Bishop2026_FrontPsychol/ (https://doi.org/10.5281/zenodo.13376172, CC BY 4.0):
  MIDI.zip      MIDI/P1.txt, MIDI/P2.txt (strikes of the two players of each dyad) and MIDI/metronome.txt (clicks of the
                distractor metronome): timestamp (ms), pitch, velocity, trial (0 = baseline), distractor_pitch, distractor_time,
                [visual_contact, rhythm_group,] dyad; trimmed by the authors to the 110-s analysis window of each trial
  Rhythms.zip   the stimulus interbeat intervals (not needed: the clicks actually played are in metronome.txt)
Motion.zip (head and hand motion capture, 353 MB) was not converted: this ZIP holds the drum-strike timing.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Bishop2026_FrontPsychol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in __import__("json").load(open(RAW / "SOURCES.json"))}
checks.internal("MIDI.zip matches the downloaded hash", stdlib.sha256(RAW / "MIDI.zip") == src["MIDI.zip"]["sha256"])
z = zipfile.ZipFile(RAW / "MIDI.zip")
T = {k: pd.read_csv(io.BytesIO(z.read(f"MIDI/{k}.txt")), sep=" ", na_values=["NA"]) for k in ("P1", "P2", "metronome")}
for k, t in T.items():
    t["trial"] = t["trial"].astype(int)
    t["dyad"] = t["dyad"].astype(int)

dyads = sorted(T["P1"].dyad.unique())
checks.internal("P1, P2 and the metronome cover the same 24 dyads (25 pairs recruited, one excluded by the authors, dyad 10)",
                dyads == sorted(T["P2"].dyad.unique()) == sorted(T["metronome"].dyad.unique()) and len(dyads) == 24 and 10 not in dyads)
tr_cond = {}  # (dyad, trial) -> (noise, pitch, visual, rhythm group) ; constant within a trial
for k in ("P1", "P2"):
    g = T[k].groupby(["dyad", "trial"]).agg(n=("distractor_time", "nunique"), p=("distractor_pitch", "nunique"),
                                              v=("visual_contact", "nunique"), r=("rhythm_group", "nunique"))
    checks.internal(f"{k}: noise type, pitch, visual contact and stimulus group are constant within each trial",
                    bool((g[["n", "p", "v", "r"]].max() <= 1).all()))
grp_of = T["P1"].dropna(subset=["rhythm_group"]).groupby("dyad").rhythm_group.agg(lambda s: int(s.iloc[0]))
checks.internal("stimulus group (1-3) is constant within each dyad", bool((T["P1"].dropna(subset=["rhythm_group"]).groupby("dyad").rhythm_group.nunique() == 1).all()))

recs = []
n_bounce = 0
for d in dyads:
    for who in ("P1", "P2"):
        t = T[who][T[who].dyad == d]
        assert t.groupby("trial", sort=False).timestamp.apply(lambda s: s.is_monotonic_increasing).all()
        t = t.copy()
        t["ibi"] = t.groupby("trial", sort=False).timestamp.diff()
        t["ev"] = t.groupby("trial", sort=False).cumcount() + 1
        t["tt"] = pd.factorize(t.trial)[0] + 1
        t["who"] = who
        recs.append(t)
tab0 = pd.concat(recs, ignore_index=True)
bounce = (tab0.ibi < 200).to_numpy()
checks.internal("strikes less than 200 ms after the player's previous strike (stick bounces, discarded by the authors)", True,
                f"{int(bounce.sum())} of {len(tab0)} strikes")
base = tab0.trial.to_numpy() == 0
noise = tab0.distractor_time.astype(str).to_numpy()
pitch = tab0.distractor_pitch.astype(str).to_numpy()
vis = tab0.visual_contact.astype(object).where(tab0.visual_contact.notna(), "").to_numpy()
cond_txt = np.where(base, "baseline, no distractor", [f"noise {a}, {'perfect' if b == 'perfect' else 'diminished'} fifth, visual contact {str(c).lower()}"
                                                    for a, b, c in zip(noise, pitch, vis)])
tab = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": "dyadic drumming with a distractor metronome",
    "group": [f"stimulus group {int(grp_of[d])}" for d in tab0.dyad],
    "subject": [f"{d}_{w}" for d, w in zip(tab0.dyad, tab0.who)], "subject_excluded_by_authors": False,
    "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
    "device": "MIDI-enabled drum pad (velocity sensitive), played in pairs; headphones with the distractor metronome",
    "session": 1, "block": np.where(base, "baseline", [f"trial {k}" for k in tab0.trial]),
    "trial": tab0.groupby(["dyad", "who"], sort=False).cumcount().to_numpy() + 1,
    "phase": np.where(base, "baseline", "test"), "time_s": tab0.timestamp.to_numpy() / 1000.0,
    "task": "finger tapping and timing", "unit": "tap", "condition": cond_txt,
    "pert_type": np.where(base, "none", "distractor_metronome"), "pert_value": np.where(base, 0.0, np.nan), "pert_unit": "",
    "feedback": "normal", "excluded_by_authors": bounce,
    "tapping_trial": tab0.tt.to_numpy(), "event_in_trial": tab0.ev.to_numpy(),
    "stimulus": np.where(base, "none", [f"{a}, {b} fifth" for a, b in zip(noise, pitch)]),
    "tap_time_ms": tab0.timestamp.to_numpy(), "stimulus_time_ms": "", "asynchrony_ms": "", "iti_ms": "", "ioi_ms": "",
    "stimulus_played": "", "strike_velocity": tab0.velocity.to_numpy(), "midi_note": tab0.pitch.to_numpy(),
})
# metronome clicks: kept as a second table in the ZIP; dyad k = the k-th dyad in the authors' numbering = persons 2k-1 and 2k
dyad_k = {d: i + 1 for i, d in enumerate(dyads)}
m = T["metronome"]
met = pd.DataFrame({"dyad": m.dyad.map(dyad_k), "trial": m.trial, "click_time_ms": m.timestamp, "midi_note": m.pitch, "velocity": m.velocity,
                    "distractor_noise": m.distractor_time, "distractor_pitch": m.distractor_pitch})
met_csv = met.to_csv(index=False, lineterminator="\r\n")

# ---------------------------------------------------------------- checks against the paper
n_dyad = tab0.dyad.nunique()
per_trial = tab0.groupby(["dyad", "trial"]).size()
trials_per_dyad = tab0.groupby("dyad").trial.nunique()
checks.published("dyads in the analysis (25 pairs, 1 excluded)", n_dyad, 24, 0, "Methods, Participants (Zenodo record description)")
checks.published("participants (two players per dyad, 24 dyads)", tab.subject.nunique(), 48, 0, "Methods, Participants (25 pairs, 1 pair excluded)")
checks.published("trials per complete dyad (12 experimental trials plus the baseline)", int(trials_per_dyad.max()), 13, 0, "Methods, Procedure")
span = tab0.groupby(["dyad", "trial"]).timestamp.agg(lambda s_: (s_.max() - s_.min()) / 1000.0)
checks.internal("median time from first to last strike of a trial is within the authors' 110-s window (not a published match: players may not strike through the whole window)", float(span.median()) <= 110.0, f"{float(span.median()):.1f} s")
checks.internal("complete dyads (all 13 trials) in the deposit", True, f"{int((trials_per_dyad == 13).sum())} of {n_dyad}")
checks.internal("trial windows: every dyad-trial lasts at most 110 s from first to last strike", float((tab0.groupby(["dyad", "trial"]).timestamp.agg(lambda s: s.max() - s.min())).max()) <= 110000)
checks.internal("12 distinct visual x noise x pitch conditions in the complete dyads' experimental trials",
                all(tab0[(tab0.dyad == d) & (tab0.trial > 0)].groupby("trial").apply(lambda g: (g.distractor_time.iloc[0], g.distractor_pitch.iloc[0], g.visual_contact.iloc[0])).nunique() == 12
                    for d in trials_per_dyad[trials_per_dyad == 13].index))
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    else:
        print(f"  [{'ok' if ch['ok'] else 'FAILED'}] {ch['what']} {ch.get('detail', '')}")

meta = dict(
    study="Bishop & Kwak 2026",
    paper=dict(authors="Bishop L, Kwak D", year=2026, title="Ignoring a noisy metronome during dyadic drumming",
               venue="Front Psychol 17:1929677", doi="10.3389/fpsyg.2026.1929677"),
    lab="Bishop (RITMO, University of Oslo)",
    data_url="https://doi.org/10.5281/zenodo.13376172",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format with dyad numbers replaced by participant codes; time stamps kept in ms and "
             "also given in seconds (time_s); the distractor metronome's clicks are in metronome_clicks.csv inside the ZIP. "
             "Not converted: the head and hand motion capture (Motion.zip) and the rhythmic stimulus files (Rhythms.zip)."),
    how=[
        "One row is one strike of one player on a MIDI drum pad: tap_time_ms is the authors' MIDI time stamp (ms; the authors "
        "trimmed each trial to a 110-s window), strike_velocity the MIDI velocity and midi_note the note number of the pad. "
        "Each dyad has a baseline trial (block baseline, no distractor) and up to 12 experimental trials, each with a "
        "distractor metronome whose timing noise (LRC, SRC or white noise), pitch (perfect or diminished fifth) and the "
        "visual contact between the players are in condition; the pair always tried to drum together at 75 bpm.",
        "Asynchronies between players and inter-strike intervals were not shared, so asynchrony_ms, iti_ms and ioi_ms are "
        "empty. excluded_by_authors marks strikes less than 200 ms after the same player's previous strike, the stick "
        "bounces that the authors discarded (Methods); group is the dyad's rhythm stimulus group (1-3).",
        "metronome_clicks.csv has one row per click of the distractor metronome (dyad k, k = 1-24 in the authors' dyad "
        "order, is persons 2k-1 and 2k of the main table). The 25th pair, which the authors excluded, is not in the deposit.",
    ],
    unit_note="drum strike",
    differences=[],
)
stdlib.release(ID, "tapping", tab, meta, checks, extra_files=[("metronome_clicks.csv", met_csv.encode("utf-8"))])
