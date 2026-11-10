"""Rossi et al. 2024 (npj Sci Learn): generalization of split-belt adaptation from treadmill to overground walking.
Deposit (Zenodo 12522690 = Dryad 10.5061/dryad.g4f4qrfwc): Dataset.zip with data/<group>.mat for five groups of 10 people.
Per group: SLA (cell per trial of participants x strides), plus overground speeds per pass, speed-match task signals and a
MATLAB 'Trials' table that cannot be read without MATLAB. Trial labels below are inferred from the paper's protocol and the
stride counts of each trial and are validated against the paper's baseline and catch-trial values."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Rossi2024_npjSciLearn"
MODULE = "walking"
DEVICE = "Instrumented split-belt treadmill and overground walkway; motion capture"
GROUPS = [("ShortPreferred", "Experiment 1A"), ("LongPreferred", "Experiment 1B"), ("ShortSlow", "Experiment 1B"),
          ("SwitchSlow", "Experiment 2"), ("SwitchControl", "Experiment 2")]
# (block label, phase, environment, belts): belts "tied", "split" or None
def layout(n_trials, group):
    L = [("OG baseline", "baseline", "OG", None)] + [(f"TM baseline {i}", "baseline", "TM", "tied") for i in range(1, 5)]
    L += [("adaptation, before catch", "learning", "TM", "split"), ("catch trial", "other", "TM", "tied"),
          ("adaptation, after catch", "learning", "TM", "split"), ("OG post-adaptation 1", "transfer", "OG", None)]
    if n_trials == 14:
        L += [(f"TM post-adaptation {i}", "washout", "TM", "tied") for i in range(1, 6)]
    elif n_trials == 12:
        L += [(f"OG post-adaptation {i}", "transfer", "OG", None) for i in range(2, 5)]
    else:
        assert n_trials == 9
    return L


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "Dataset.zip")
    frames, per_group = [], {}
    for g, exp in GROUPS:
        m = sio.loadmat(io.BytesIO(z.read(f"Dataset/data/{g}.mat")), squeeze_me=False, struct_as_record=False)
        s = m[g][0, 0]
        sla = list(s.SLA.ravel())
        lay = layout(len(sla), g)
        assert len(lay) == len(sla), (g, len(sla))
        ns = sla[0].shape[0]
        per_group[g] = dict(sla=sla, n=ns, lay=lay)
        for p in range(ns):
            parts, k = [], 0
            for t, (blk, phase, env, belts) in enumerate(lay):
                v = sla[t][p].astype(float)
                alive = np.isfinite(v)
                n = int(alive.sum())
                assert alive[:n].all(), (g, p, t)
                v = v[:n]
                split = belts == "split"
                parts.append(pd.DataFrame(dict(
                    dataset=DS, study=DS, experiment=exp, group=g, subject=f"{g}_{p + 1}", subject_excluded_by_authors=False,
                    population="healthy", setting="lab", device=DEVICE, session=1, block=blk, phase=phase, task="walking",
                    unit="stride", condition="overground" if env == "OG" else "treadmill",
                    pert_type="split_belt" if split else "none", pert_value=3.0 if split else 0.0, pert_unit="ratio" if split else "",
                    feedback="normal", excluded_by_authors=False, fast_leg="right",
                    belt_speed_fast_mps=(1.5 if split else 0.5) if belts else np.nan, belt_speed_slow_mps=0.5 if belts else np.nan,
                    step_length_asymmetry=v, walking_context="overground" if env == "OG" else "treadmill", walking_trial=t + 1,
                ), index=range(k, k + n)))
                k += n
            d = pd.concat(parts, ignore_index=True)
            d["trial"] = np.arange(1, len(d) + 1)
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["step_length_asymmetry"] = df.step_length_asymmetry.round(6)

    sp = df[df.group == "ShortPreferred"]

    def mean_of_person_means(d, first=None):
        v = [(x.step_length_asymmetry.iloc[:first] if first else x.step_length_asymmetry).mean() for _, x in d.groupby("subject", sort=False)]
        return float(np.mean(v))

    ck.internal("5 groups of 10 participants each, 50 in all (paper: 50 neurotypical adults)",
                all(v["n"] == 10 for v in per_group.values()) and df.subject.nunique() == 50)
    ck.internal("trial counts per group are 14 (Short/LongPreferred, ShortSlow), 12 (SwitchSlow) and 9 (SwitchControl)",
                [len(v["lay"]) for v in per_group.values()] == [14, 14, 14, 12, 9])
    ck.internal("adaptation starts negative (right/fast step shorter) and the catch trial and first post-adaptation strides are positive",
                df[(df.block == "adaptation, before catch")].groupby("subject").head(5).step_length_asymmetry.mean() < -0.2
                and df[df.block == "catch trial"].step_length_asymmetry.mean() > 0.3)
    ck.published("overground baseline SLA, Short Preferred, mean of participants' means (-0.0110)",
                 mean_of_person_means(sp[sp.block == "OG baseline"]), -0.0110, 0.0005, "Results, Experiment 1A baseline")
    tmb = sp[sp.block.str.startswith("TM baseline")]
    ck.published("treadmill baseline SLA, Short Preferred, mean of participants' means over the four baseline blocks (-0.0145)",
                 mean_of_person_means(tmb), -0.0145, 0.0005, "Results, Experiment 1A baseline")
    ck.published("treadmill catch trial SLA, Short Preferred, first 5 strides (0.9203)",
                 mean_of_person_means(sp[sp.block == "catch trial"], first=5), 0.9203, 0.0005, "Results, Experiment 1A catch trial")
    ck.published("participants (50)", df.subject.nunique(), 50, 0, "Methods, Participants")
    for c in ck.items:
        print(c)
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in df.columns and c != "person":
            df[c] = np.nan

    meta = dict(
        study=DS,
        paper=dict(authors="Rossi C, Roemmich RT, Bastian AJ", year=2024, title="Understanding mechanisms of generalization following locomotor adaptation",
                   venue="npj Sci Learn 9(1):48", doi="10.1038/s41539-024-00258-2"),
        lab="Bastian (Kennedy Krieger / Johns Hopkins)",
        data_url="https://zenodo.org/records/12522690 (same files as Dryad https://doi.org/10.5061/dryad.g4f4qrfwc)",
        license="CC0 1.0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Kept the authors' per-stride step length asymmetry (SLA) of the five group files, with block labels, belt speeds and "
                 "participant codes added from the paper; the deposit's joint-angle and marker values at gait events (ShortPreferred only), "
                 "interpolated angle data, overground speeds per pass, speed-match task signals, questionnaire answers, bootstrap files and "
                 "analysis code are not included."),
        how=["One row is one stride of one participant, in the deposit's order within each group's trial sequence: overground baseline, "
             "four treadmill baseline blocks (belts tied at 0.5 m/s), split-belt adaptation (right belt fast at 1.5 m/s, left belt slow at 0.5 m/s) "
             "interrupted by a 10 s tied-belt catch trial, overground post-adaptation block 1, then either five treadmill tied-belt post-adaptation "
             "trials (Experiment 1 groups) or overground blocks 2 to 4 (SwitchSlow); SwitchControl ends after overground block 1.",
             "step_length_asymmetry is the authors' SLA: (right step length - left step length) / sum, with the right leg the fast leg; "
             "negative early in adaptation and strongly positive in the catch trial. walking_trial is the trial's position in the group's sequence.",
             "The deposit's trial table is a MATLAB table object that cannot be read here, so each trial's label (which trial is the catch trial, "
             "overground or treadmill) is inferred from the paper's protocol, the number of strides in each trial and the presence of overground speeds; "
             "the paper's baseline and catch-trial values for Short Preferred are reproduced exactly under this reading. Ages and sexes are in the paper only as totals.",
             "SwitchSlow is the paper's Switch group (slow overground walking); SwitchControl is its Reference group (matching file names to the paper's group "
             "names is an inference)."],
        unit_note="stride",
        differences=["The paper gives group sizes only as the bootstrap size of 10; the files hold 10 participants each."],
    )
    stdlib.release(DS, MODULE, df, meta, ck)


if __name__ == "__main__":
    main()
