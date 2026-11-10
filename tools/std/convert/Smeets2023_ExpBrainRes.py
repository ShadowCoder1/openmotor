"""Smeets et al. (2022), Exp Brain Res 241:105-111 (listed as 2023): manual size indication after opposite prism adaptation of the thumb
and the index finger, 12 participants, one Optotrak recording (250 Hz, 3 s, two markers on the nails of thumb and index finger) per trial.
One row per recording.

Run from the site folder: python3 -I tools/std/convert/Smeets2023_ExpBrainRes.py
Raw files (untrusted data, read only): raw_extra/Smeets2023_ExpBrainRes/ Individual_raw.zip (dataPP1..12 with TN######.ndf Optotrak files and
PP.mat condition labels), Output1.mat is not used; the authors' scripts (Verwerk_raw.m, Verwerk_2nd.m, readndf.m) are read as text only.
PP.mat holds a MATLAB string array (210 labels). It is decoded by Smeets2023_ExpBrainRes_mcos.py (the Sadeghi2024 reader with string_object()
extended from one string to arrays). Labels: k/g = small cube / large cuboid with target position 1-3 (size indication trials); d/w = thumb
('duim') / index finger ('wijsvinger') pointing trials with the target seen through a prism (adaptation trials).
"""
import io
import os
import re
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Smeets2023_ExpBrainRes_mcos import MatObjects, string_object  # noqa: E402

ID = "Smeets2023_ExpBrainRes"
RAW = stdlib.raw_dir(ID)
ZIP = RAW / "Individual_raw.zip"
checks = stdlib.Checks()
STUDY = "Smeets et al. 2022, Exp Brain Res"
SDIR = RAW / "_samples"
SDIR.mkdir(exist_ok=True)
FS = 250
DEVICE = "Optotrak 3020, infrared diodes on the nails of the right thumb and index finger (250 Hz); PLATO shutter glasses with 10-diopter prisms"
DL_FIRST = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1]  # the authors' Verwerk_raw.m: for each participant, whether thumb-left adaptation came first

z = zipfile.ZipFile(ZIP)
names = {i.filename for i in z.infolist() if "__MACOSX" not in i.filename}
tmpd = tempfile.mkdtemp(dir=str(RAW))


def readndf(buf):
    """Optotrak NDF, as the authors' readndf.m: 256-byte header, then float32 frames of items x 3 coordinates; values below -1e21 are missing."""
    items, sub = struct.unpack_from("<hh", buf, 1)
    nfr = struct.unpack_from("<i", buf, 5)[0]
    fq = struct.unpack_from("<f", buf, 9)[0]
    assert sub == 3 and fq == FS, (sub, fq)
    a = np.frombuffer(buf, dtype="<f4", count=items * sub * nfr, offset=256).astype(float).reshape(nfr, items * sub)
    a[a < -10e20] = np.nan
    return items, a


def blocks_for(pp, n_files):
    """Files (1-based, in file-name order) of the five phases, from the authors' Verwerk_raw.m: pre 1-30, post-adaptation blocks of 30 trials
    starting at the participant-specific file numbers they list (the default 91 and 181); the files in between are the two adaptation phases."""
    p1 = {5: (90, 119), 9: (91, 119), 10: (114, 143)}.get(pp, (91, 120))
    p2 = {10: (204, 233), 12: (180, 209)}.get(pp, (181, 210))
    return (1, 30), p1, p2


rows, extra = [], []
aper = {}  # (pp, phase, size) -> list of aperture values in mm (the authors' last-0.5-s measure)
n_files_total, ad_unlabelled = 0, 0
for pp in range(1, 13):
    tmp_mat = os.path.join(tmpd, f"PP{pp}.mat")
    open(tmp_mat, "wb").write(z.read(f"dataPP{pp}/PP.mat"))
    o = MatObjects(tmp_mat)
    lab = string_object(o, sio.loadmat(tmp_mat, squeeze_me=True, struct_as_record=False)["PP"])
    assert len(lab) == 210 and all(re.fullmatch(r"[kgdw][123]", s) for s in lab), lab[:5]
    files = sorted(n for n in names if re.fullmatch(rf"dataPP{pp}/TN\d+\.ndf", n))
    pre, p1, p2 = blocks_for(pp, len(files))
    blk = {}
    for j in range(1, len(files) + 1):
        if pre[0] <= j <= pre[1]:
            blk[j] = ("pre", lab[j - 1])
        elif p1[0] <= j <= p1[1]:
            k = j - p1[0]
            if pp == 9:  # the authors drop label 94: this participant has no recording for that trial
                idx = [91, 92, 93] + list(range(95, 121))
                blk[j] = ("post1", lab[idx[k] - 1])
            else:
                blk[j] = ("post1", lab[90 + k])
        elif p2[0] <= j <= p2[1]:
            blk[j] = ("post2", lab[180 + j - p2[0]])
        elif pre[1] < j < p1[0]:
            blk[j] = ("ad1", None)
        else:
            blk[j] = ("ad2", None)
    for ph, (a, b), lo in (("ad1", (pre[1] + 1, p1[0] - 1), 30), ("ad2", (p1[1] + 1, p2[0] - 1), 120)):
        js = [j for j in blk if blk[j][0] == ph]
        assert js == list(range(a, b + 1))
        if len(js) == 60:
            for k, j in enumerate(js):
                blk[j] = (ph, lab[lo + k])
        else:
            ad_unlabelled += len(js)
    dl1 = bool(DL_FIRST[pp - 1])
    for j, fn in enumerate(files, 1):
        n_files_total += 1
        items, arr = readndf(z.read(fn))
        assert items == 2 and len(arr) == 750
        ph, label = blk[j]
        code = f"{ID}_S{pp:04d}"
        sample = f"samples/{code}_trial{j:03d}.csv"
        df = pd.DataFrame({"time_s": np.arange(750) / FS, "thumb_x": arr[:, 0], "thumb_y": arr[:, 1], "thumb_z": arr[:, 2],
                           "index_x": arr[:, 3], "index_y": arr[:, 4], "index_z": arr[:, 5]})
        local = os.path.join(SDIR, f"{code}_trial{j:03d}.csv")
        df.to_csv(local, index=False, float_format="%.8g", lineterminator="\r\n", na_rep="")
        extra.append((sample, local))
        info, ev = [], ""
        size = None
        if ph in ("pre", "post1", "post2"):
            size = "small" if label[0] == "k" else "large"
            d = arr[:, 3:6] - arr[:, 0:3]
            af = float(np.mean(np.sqrt((d ** 2).sum(axis=1))[624:750]))  # mean of the last 0.5 s of the 3 s recording, as Verwerk_raw.m
            aper.setdefault((pp, ph, size), []).append(af)
            info = [f"target_size={'small cube' if size == 'small' else 'large cuboid'}", f"target_position={label[1]}",
                    f"hand_opening_last_0p5s_mm={'' if np.isnan(af) else format(af, '.8g')}"]
            cond = f"{'small cube' if size == 'small' else 'large cuboid'} at position {label[1]}"
            pert_t, pert_v, pert_u, fb = "none", 0, "", "normal"
            if ph == "pre":
                bl, phase = "pre-adaptation", "baseline"
            else:
                first = (ph == "post1")
                tl = dl1 if first else not dl1
                bl, phase = f"post-adaptation {1 if first else 2}", "aftereffect"
                info.append("preceding_adaptation=" + ("thumb_left_index_right" if tl else "thumb_right_index_left"))
        else:
            first = (ph == "ad1")
            tl = dl1 if first else not dl1
            bl, phase = f"adaptation {1 if first else 2}", "learning"
            pert_t, pert_v, pert_u, fb = "prism", 5, "cm", "altered"
            info = ["pairing=" + ("thumb_left_index_right" if tl else "thumb_right_index_left")]
            if label is None:
                cond = "pointing trial (digit not recoverable)"
            else:
                dig = "thumb" if label[0] == "d" else "index finger"
                cond = f"pointing with the {dig}"
                info += [f"digit={dig.replace(' ', '_')}", f"target_position={label[1]}"]
        rows.append(dict(
            dataset=ID, study=STUDY, experiment="Manual size indication after prism adaptation", group="",
            subject=str(pp), person=str(pp), subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy",
            setting="lab", device=DEVICE, session=1, block=bl, trial=j, phase=phase, time_s="", task="reaching", unit="recording",
            condition=cond, pert_type=pert_t, pert_value=pert_v, pert_unit=pert_u, feedback=fb, excluded_by_authors=False,
            recording=j, samples_file=sample, signal_kind="3D positions of two nail markers (thumb, index finger)",
            channels="thumb_x;thumb_y;thumb_z;index_x;index_y;index_z", channel_units="mm;mm;mm;mm;mm;mm",
            sample_rate_hz=FS, n_samples=750, duration_s=750 / FS, events="",
            processing="none (raw reconstructed Optotrak coordinates; missing samples left empty)", trial_info=";".join(info)))
D = pd.DataFrame(rows)

# ---- the authors' outcome measures per participant (Verwerk_raw.m / Verwerk_2nd.m), in cm
def pm(pp, ph, size):
    v = np.array(aper[(pp, ph, size)], dtype=float)
    return np.nanmean(v)  # rmmissing in the authors' script


size_eff, after = [], []
for pp in range(1, 13):
    dl1 = bool(DL_FIRST[pp - 1])
    TL, TR = ("post1", "post2") if dl1 else ("post2", "post1")  # post-adaptation block after thumb-left/index-right adaptation, and after the opposite
    big = np.mean([pm(pp, "pre", "large"), pm(pp, TL, "large"), pm(pp, TR, "large")])
    small = np.mean([pm(pp, "pre", "small"), pm(pp, TL, "small"), pm(pp, TR, "small")])
    size_eff.append((big - small) / 10)
    after.append(((pm(pp, TR, "small") - pm(pp, TL, "small")) + (pm(pp, TR, "large") - pm(pp, TL, "large"))) / 2 / 10)
size_eff, after = np.array(size_eff), np.array(after)
t_after = stats.ttest_1samp(after, 0)

P = checks.published
P("participants", D.person.nunique(), 12, 0, "Methods, Participants (twelve volunteers)")
P("size scaling: mean difference between large and small target (cm)", size_eff.mean(), 2.5, 0.05, "Results (2.5 +- 0.7 cm)")
P("size scaling: SD across participants (cm)", size_eff.std(ddof=1), 0.7, 0.05, "Results (2.5 +- 0.7 cm)")
P("aftereffect mean (cm)", after.mean(), -0.1, 0.05, "Results (-0.1 +- 0.2 cm)")
P("aftereffect SD (cm)", after.std(ddof=1), 0.2, 0.05, "Results (-0.1 +- 0.2 cm)")
P("recording frequency (Hz)", FS, 250, 0, "Setup and procedure (250 Hz)")
P("size-indication trials per phase (2 sizes x 3 positions x 5)", D[D.phase.isin(["baseline"])].groupby("person").size().mean(), 30, 0, "Setup and procedure (30 size indication trials)")
checks.internal("every participant has 30 pre-adaptation recordings", (D[D.phase == "baseline"].groupby("person").size() == 30).all())
checks.internal("every participant has two post-adaptation blocks of 30 recordings (29 for one participant with a missing trial)",
                (D[D.phase == "aftereffect"].groupby("person").size().isin([59, 60])).all())
checks.internal("2542 recordings = every ndf file outside the authors' 'Foute data' (bad data) folders", len(D) == 2542 and n_files_total == 2542, str(len(D)))
checks.internal("every recording has 750 samples and a samples file", (D.n_samples == 750).all() and len(extra) == len(D))

how = [
    "One row is one 3 s Optotrak recording (250 Hz, 750 samples) of one trial: the 3D positions of a marker on the nail of the thumb and one on the index finger (mm). "
    "Three kinds of trial: size indication before and after adaptation (the participant opens the unseen hand to show the size of a small cube or a twice as large cuboid and holds it for 2 s), "
    "and pointing trials during the two adaptation phases (touching the target with the thumb or with the index finger, seen monocularly through a 10-diopter prism, about 5 cm of displacement; "
    "the prism orientation depends on the digit, and the pairing is swapped between the two phases).",
    "Samples (samples/<subject>_trial<nnn>.csv): time_s from the start of the recording, then thumb_x, thumb_y, thumb_z, index_x, index_y, index_z in mm. Samples the system could not reconstruct are empty. "
    "Which marker is the thumb and which the index finger is taken from the authors' script (first marker 'D', duim = thumb; second marker 'W', wijsvinger = index finger). "
    "Axis directions of the Optotrak frame are not documented in the deposit.",
    "Trial labels come from PP.mat (a MATLAB string array, read without MATLAB): k = small cube, g = large cuboid, with target position 1-3 (size indication trials); d = thumb, w = index finger, "
    "with target position 1-3 (pointing trials). Recordings are numbered in file-name order. Phase boundaries are those of the authors' Verwerk_raw.m: 30 pre-adaptation recordings, then adaptation 1, "
    "30 post-adaptation 1, adaptation 2 and 30 post-adaptation 2 (block starts differ for participants 5, 9, 10 and 12 in the authors' script and are used as they give them). "
    "For the pointing trials the label is attached only where the phase has the planned 60 recordings; otherwise the condition says the digit is not recoverable.",
    "For the size indication trials trial_info holds the authors' outcome measure hand_opening_last_0p5s_mm: the mean distance between the two markers over the last 0.5 s of the recording "
    "(samples 625-750), as computed in their script. 'preceding_adaptation' / 'pairing' gives the authors' thumb-left/index-right or thumb-right/index-left pairing (their DL_first list).",
]
meta = dict(
    paper=dict(authors="Smeets JBJ, Pennekamp I, van Amsterdam B, Schot WD", year=2022,
               title="How prism adaptation reveals the distinct use of size and positions in grasping", venue="Exp Brain Res 241(1):105-111",
               doi="10.1007/s00221-022-06506-4"),
    lab="Smeets & Brenner (VU Amsterdam)", data_url="https://osf.io/cm8wy/", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Each .ndf Optotrak file was converted to one samples CSV (coordinates unchanged, missing samples empty); the trial labels in PP.mat were decoded and attached. The 13 recordings the authors "
             "moved to their 'Foute data' (bad data) folders are not included. Output1.mat (compiled means), the .txt Optotrak settings files, params.old and the authors' MATLAB scripts are not included. "
             "The paper lists no ages per participant (19-21 years, 6 men and 6 women), so age and sex are empty."),
    how=how,
    differences=[],
    unit_note="recording (one 3 s Optotrak trial)",
)
meta["differences"] = [
    "For participants 5, 9, 10 and 12 the number of recordings differs from the planned 210 (participant 10 has 23 extra recordings in the first adaptation phase, participant 12 has 209), "
    "and the authors' Verwerk_raw.m uses participant-specific start files for the post-adaptation blocks; this conversion follows those. "
    "In the authors' script the 30th slot of participant 9's first post-adaptation block is not reset and so carries a trial of the previous participant; here that block has the 29 real recordings, "
    "so participant 9's aftereffect is -0.004 cm here against 0.008 cm in the authors' compiled Output1.mat (all other participants agree to 3 decimals). "
    "As a result the one-sample t of the aftereffect is -1.688 here (one-sided p 0.940) against t(11) = -1.668 (p = 0.938) in the paper; these two are therefore not used as checks."]
stdlib.release(ID, "timeseries", D, meta, checks, extra_files=extra)
