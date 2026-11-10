"""Schmitter & Straube (2022), Front Hum Neurosci 16:998843: cerebellar tDCS and sensorimotor / inter-sensory temporal recalibration,
one row per delay-detection test trial (module perception).

Run from the site folder: python3 -I tools/std/convert/Schmitter2022_FrontHumNeurosci.py
Raw file (untrusted data, read only): raw_extra/Schmitter2022_FrontHumNeurosci/data.zip, from https://zenodo.org/records/6861087:
  data/trialwise_data_long.csv  (';' separated, decimal comma) one row per test trial; readme.docx (read as text) documents the columns
  except RT. The other tables (psychometric thresholds, TRE, side-effect ratings) are derived summaries and are not used.
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Schmitter2022_FrontHumNeurosci"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

src = {r["file"]: r for r in __import__("json").load(open(RAW / "SOURCES.json"))}
checks.internal("data.zip matches the downloaded hash", stdlib.sha256(RAW / "data.zip") == src["data.zip"]["sha256"])
z = zipfile.ZipFile(RAW / "data.zip")
d = pd.read_csv(io.BytesIO(z.read("data/trialwise_data_long.csv")), sep=";", decimal=",", encoding="utf-8-sig")
n = len(d)

# session = order in which the participant's four stimulation conditions appear in the file (blocks of 384 consecutive rows)
runs = []
for p, g in d.groupby("participant", sort=False):
    st = g.stimulation.tolist()
    order = [s for i, s in enumerate(st) if i == 0 or s != st[i - 1]]
    checks.internal(f"participant {p}: the four stimulation conditions each form one block of rows", len(order) == 4 and len(set(order)) == 4)
    runs.append(g.stimulation.map({s: i + 1 for i, s in enumerate(order)}))
d["session"] = pd.concat(runs).loc[d.index]

mod = d.test_modality.str.lower()
t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": ID, "group": "", "subject": d.participant.astype(str), "person": d.participant.astype(str),
    "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "right", "population": "healthy", "setting": "lab",
    "device": "button press (own finger, or moved by an electromagnetic button device) with delayed visual or auditory stimulus; cerebellar tDCS",
    "session": d.session, "block": d.stimulation + "; " + d.movement_type.str.lower() + "; " + mod + " test; adaptation " + d.adaptation_delay,
    "trial": d.trial, "phase": "test", "time_s": "", "task": "finger tapping and timing", "unit": "trial",
    "condition": d.movement_type.str.lower() + " movement, " + mod + " test, adaptation delay " + d.adaptation_delay,
    "pert_type": "delay", "pert_value": d.delay, "pert_unit": "ms", "feedback": ["altered" if x > 0 else "normal" for x in d.delay],
    "excluded_by_authors": (d.button_pressed == 0) | d.responses.isna(),
    "stimulus": "delay inserted between the button press and the " + mod + " stimulus (0 = none)",
    "stimulus_value": d.delay, "stimulus_unit": "ms", "reference_value": 0,
    "response": d.responses.map({0.0: "0", 1.0: "1"}).fillna(""), "response_label": d.responses.map({0.0: "no delay", 1.0: "delay"}).fillna(""),
    "response_value": "", "response_unit": "", "response_correct": "", "reaction_time_s": d.RT, "bout": "",
})

# ---- published numbers (Methods: participants, design, exclusions)
checks.published("participants", d.participant.nunique(), 22, 0, "Methods, Participants")
checks.published("test trials per participant (4 stimulations x 8 conditions x 8 pairs x 6 delay levels)", n / 22, 4 * 8 * 8 * 6, 0, "Methods, Design")
checks.published("delay levels", d.delay.nunique(), 6, 0, "Methods, Procedure")
checks.published("percent of trials with no button press", 100 * (d.button_pressed == 0).mean(), 0.8, 0.1, "Methods, Data analysis")
checks.published("percent of trials with a button press but no response", 100 * ((d.button_pressed == 1) & d.responses.isna()).mean(), 1.4, 0.1,
                 "Methods, Data analysis")
checks.internal("48 test trials per participant, stimulation and condition",
                set(d.groupby(["participant", "stimulation", "movement_type", "test_modality", "adaptation_delay"]).size()) == {48})
checks.internal("every row kept, in the authors' order", len(t) == n == 33792)
checks.internal("response times between 0 and 1.5 s (within the 1500 ms response window), apart from one -0.1 ms value",
                d.RT.min() > -0.001 and d.RT.max() < 1.51)

meta = dict(
    paper=dict(authors="Schmitter CV, Straube B", year=2022,
               title="The impact of cerebellar transcranial direct current stimulation (tDCS) on sensorimotor and inter-sensory temporal recalibration",
               venue="Front Hum Neurosci 16:998843", doi="10.3389/fnhum.2022.998843"),
    lab="Straube (University of Marburg)", data_url="https://zenodo.org/records/6861087",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format from trialwise_data_long.csv. The psychometric-function thresholds and slopes, the temporal "
             "recalibration effects and the side-effect ratings are derived tables and are not included. Session is added from the order of the "
             "stimulation blocks in the file."),
    how=["One row is one test trial of a delay-detection task. In an adaptation phase the participant pressed a button 18 times (own finger: active, or moved by "
         "an electromagnetic device: passive) and each press produced a visual or auditory stimulus after 0 or 150 ms; in the test phase a button press "
         "produced the stimulus after one of six delays (0, 83, 167, 250, 333, 417 ms) and the participant answered whether they saw or heard a delay. "
         "Each participant had four sessions, one per cerebellar tDCS condition (anodal, cathodal, dual-hemisphere, sham); each session had eight conditions "
         "(active or passive, visual or auditory, adaptation delay 0 or 150 ms), each of 8 adaptation-test pairs with 6 test trials.",
         "delay (pert_value and stimulus_value) is in ms. response is 1 if the participant said there was a delay and 0 if not; empty when no response was "
         "registered. reaction_time_s is the deposit's RT column (seconds), which the readme does not describe; its range (0 to 1.5 s, matching the 1500 ms "
         "response window of the paper) suggests the time from the response prompt to the answer. Rows without a button press or without a response are "
         "kept and marked excluded_by_authors, as the paper excludes them. The tDCS condition is part of block and condition (it is not a perturbation of "
         "the movement). Session numbers follow the order of the four stimulation blocks in the authors' file; the deposit does not state the real session order.",
         "No age or sex per participant is shared."],
    unit_note="delay-detection test trial",
    differences=["The paper describes the outcome as the proportion of detected delays and psychometric thresholds; thresholds are not recomputed here. "
                 "The paper reports no reaction times."],
)
stdlib.release(ID, "perception", t, meta, checks)
