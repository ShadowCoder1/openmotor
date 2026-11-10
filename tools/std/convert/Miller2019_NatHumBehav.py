"""Miller (2019), Nat Hum Behav 3(11):1225-1235, ECoG library: the finger flexion experiment (fingerflex.zip, 9 patients), data glove part only.
One row per patient: finger flexion of the five digits (data glove, as shared), the on-screen cue and the labelled movement epochs; the
ECoG channels (neural data) are not carried. Task: cued, self-paced flexion of one finger at a time in 2-s movement trials with 2-s rests
(Miller et al. 2012 PLoS Comput Biol 8(9):e1002655, the paper the experiment was first published in; read from the deposit's readme).
Deposit: https://purl.stanford.edu/zk881ps0522 (CC BY-SA 4.0), file fingerflex.zip read in place (no extraction).
Run from the site folder: python3 -I tools/std/convert/Miller2019_NatHumBehav.py"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Miller2019_NatHumBehav"
raw = stdlib.raw_dir(DS)
ck = stdlib.Checks()
z = zipfile.ZipFile(raw / "fingerflex.zip")
# readme table (README_fingerflex_dataset_notes.docx): code, subject number, age, sex, handedness, array location
DEMO = [("bp", 1, 18, "F", "R", "left frontoparietal"), ("cc", 2, 21, "M", "R", "right frontotemporal"),
        ("zt", 3, 27, "F", "R", "left fronto-temporo-parietal"), ("jp", 4, 35, "F", "R", "left fronto-temporal"),
        ("ht", 5, 26, "M", "R", "left parieto-temporo-occipital"), ("mv", 6, 45, "F", "R", "left frontotemporal"),
        ("wc", 7, 32, "M", "R", "left fronto-temporo-parietal"), ("wm", 8, 19, "F", "R", "right fronto-parietal"),
        ("jc", 9, 18, "F", "R", "left frontal")]
FN = ["thumb", "index", "middle", "ring", "little"]
rows, files, part = [], [], []
seen_codes, stim_codes = set(), set()
for code, n, age, sex, hand, loc in DEMO:
    pid = f"P{n}"
    m = sio.loadmat(io.BytesIO(z.read(f"fingerflex/data/{code}/{code}_fingerflex.mat")), variable_names=["flex", "cue"])
    st = sio.loadmat(io.BytesIO(z.read(f"fingerflex/data/{code}/{code}_stim.mat")))["stim"].ravel()
    flex, cue = m["flex"], m["cue"].ravel()
    assert flex.shape[1] == 5 and len(cue) == len(flex) == len(st)
    seen_codes |= set(np.unique(cue).astype(int))
    stim_codes |= set(np.unique(st).astype(int))
    fn = f"samples/{DS}_{pid}.csv"
    ntr = int(((np.diff(np.r_[0, cue]) != 0) & (cue > 0)).sum())
    rows.append(dict(
        dataset=DS, study="Miller 2019 Nat Hum Behav (ECoG library, finger flexion experiment)", experiment="fingerflex", group="", subject=pid,
        person=pid, subject_excluded_by_authors=False, age=float(age), sex=sex, handedness=hand, time_s="", population="epilepsy patients with implanted ECoG",
        setting="hospital bedside", device="data glove (five digits), 1000 Hz alignment with ECoG", session=1, block="", trial=1, phase="test",
        task="finger flexion", unit="recording", condition="cued self-paced flexion of one finger per 2-s trial, 2-s rests", pert_type="none",
        pert_value=0.0, pert_unit="", feedback="normal", excluded_by_authors=False, recording=1, samples_file=fn,
        signal_kind="data glove flexion of thumb, index, middle, ring and little finger, with the screen cue and labelled movement epochs",
        channels="flex_thumb;flex_index;flex_middle;flex_ring;flex_little;cue;stim", channel_units="raw glove units (as shared);raw glove units (as shared);"
        "raw glove units (as shared);raw glove units (as shared);raw glove units (as shared);code 0-5;code 0-5",
        sample_rate_hz=1000, n_samples=len(cue), duration_s=len(cue) / 1000, events="",
        processing=("none: glove values are in raw form as shared, updated in 40-ms blocks and with a non-zero offset (readme); ECoG channels of the "
                    "deposit are not included"), trial_info=f"cue_epochs={ntr};electrode_array={loc}"))
    files.append((fn, np.column_stack([flex, cue, st])))
df = pd.DataFrame(rows)
part = [dict(person=f"P{n}", electrode_array=loc) for _, n, _, _, _, loc in DEMO]


def extra():
    for fn, a in files:
        yield fn, "time_s," + ",".join(f"flex_{f}" for f in FN) + ",cue,stim\n" + "".join(
            f"{j / 1000:.3f}," + ",".join(format(x, ".8g") for x in a[j]) + "\n" for j in range(len(a)))


ck.published("patients in the finger flexion experiment", len(df), 9, 0, "Miller 2019 / Miller 2012: nine patients (deposit readme subject table)")
ck.published("distinct finger codes of the cue", len(seen_codes - {0}), 5, 0, "readme: cue codes 1-5 = thumb to little finger")
ck.published("digits recorded by the glove (columns of flex)", flex.shape[1], 5, 0, "readme: flex has five columns thumb-index-middle-ring-little")
ck.internal("cue codes are 0-5 only", seen_codes <= set(range(6)))
ck.internal("stim codes outside the readme list are -1 and -2 only", stim_codes - set(range(6)) <= {-1, -2}, str(sorted(stim_codes)))
ck.internal("one sample file per patient", df.samples_file.is_unique and len(files) == 9)
ck.internal("recording lengths plausible (over 2 minutes)", bool((df.duration_s > 120).all()), str(df.duration_s.round(0).tolist()))

meta = dict(
    paper=dict(authors="Miller KJ", year=2019, title="A library of human electrocorticographic data and analyses", venue="Nat Hum Behav 3(11):1225-1235",
               doi="10.1038/s41562-019-0678-3"),
    lab="Miller (Stanford / Mayo Clinic)", data_url="https://purl.stanford.edu/zk881ps0522", license="CC BY-SA 4.0",
    license_url="https://creativecommons.org/licenses/by-sa/4.0/",
    changes=("From the finger flexion experiment of the library (fingerflex.zip) only the behavioural variables were kept: the five data-glove flexion "
             "channels (flex), the screen cue (cue) and the labelled movement epochs (stim), one CSV per patient with a time column added (sample index / 1000 "
             "Hz). The ECoG channels, electrode positions and brain surfaces are not included (neural data). Patients are coded by the table order of the readme "
             "(the authors' two-letter patient codes are not used). The ethics statement the data owners require in methods sections: 'All patients participated "
             "in a purely voluntary manner, after providing informed written consent, under experimental protocols approved by the Institutional Review Board "
             "of the University of Washington (#12193). All patient data was anonymized according to IRB protocol, in accordance with HIPAA mandate.'"),
    how=["One row is one patient's finger flexion session (about 3 to 10 minutes). Samples are at 1000 Hz, time-aligned with the ECoG recording. The patient was "
         "cued by a word on a bedside monitor to move one finger during 2-s movement trials and moved that finger self-paced, typically 2 to 5 times per trial, "
         "followed by 2-s rest trials with a blank screen.",
         "flex_* are the data-glove readings of the thumb, index, middle, ring and little finger in the raw form of the deposit: in 40-ms blocks, with a non-zero "
         "offset, in unspecified glove units (larger values are not stated to mean flexion in the readme). cue is the code of the cue on the screen at each "
         "sample and stim the labelled epoch enclosing the actual movement; both use 0 = inter-stimulus interval, 1 = thumb, 2 = index, 3 = middle, 4 = ring, 5 = little finger.",
         "The task part of the paper listed for this dataset is the library paper; the experiment was first published in Miller et al. 2012 (PLoS Comput Biol 8(9):e1002655). "
         "participants.csv holds the electrode array location; age, sex and handedness are in the main table (from the readme's patient table)."],
    unit_note="data-glove session recording of one patient (samples in samples/)",
    differences=["The stim variable also holds the codes -1 (about a quarter of the samples in most patients) and -2 (a few patients, few samples), which the readme does not define; they are kept as shared. The cue of one patient is 1 (thumb) for almost the whole session and 0 otherwise, unlike the other patients' cycling cues (the deposit gives no explanation).",
                 "Published statistics of the library paper concern ECoG analyses and were not reproduced; the checks are the participant count and the code lists of the readme.",
                 "The deposit's other task zips (for example motor_basic, gestures, joystick_track, mouse_track) and the BCI Competition IV version of the finger "
                 "flexion data were not converted."],
)
pcols = [("electrode_array", "text", "", "Location of the implanted electrode array (readme table).")]
stdlib.release(DS, "timeseries", df, meta, ck, participants=part, participant_columns=pcols, extra_files=extra())
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
