"""Rossi et al. 2021 (Front Aging Neurosci): 2-back dual task during split-belt adaptation, young and late middle-aged adults.
Deposit (Zenodo 5655845 = Dryad 10.5061/dryad.18931zcxm): four MOTOR .mat files (10 participants each) holding, per
participant, phase and stride, the fast and slow step lengths (FSL, SSL, mm), double support times (DSF, DSS), limb
excursions (LEF, LES, mm), the authors' step length asymmetry (SLA) and a dual-task flag (DT). Cognitive 2-back responses are
a different kind of data (button presses) and are not part of this walking table."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Rossi2021_FrontAgingNeurosci"
MODULE = "walking"
FILES = [("YAST", "Experiment 1", "YASingle", "young adults"), ("YADT", "Experiment 1", "YADual", "young adults"),
         ("OAST", "Experiment 2", "MASingle", "late middle-aged adults"), ("OADT", "Experiment 2", "MADual", "late middle-aged adults")]
BLOCKS = ["baseline", "adaptation", "washout 1", "washout 2", "washout 3", "washout 4", "readaptation"]
PHASE = ["baseline", "learning", "washout", "washout", "washout", "washout", "learning"]
SPLIT = [False, True, False, False, False, False, True]
DEVICE = "Instrumented split-belt treadmill; Optotrak motion capture (100 Hz); 2-back letter task in some blocks"


def main():
    ck = stdlib.Checks()
    frames, sla_by = [], {}
    nsub = {}
    maxdiff = 0.0
    for code, exp, grp, pop in FILES:
        m = sio.loadmat(stdlib.raw_dir(DS) / f"{code}_MOTOR.mat")
        arr = {k: m[k].astype(float) for k in ["FSL", "SSL", "DSF", "DSS", "LEF", "LES", "SLA", "DT"]}
        ns = arr["FSL"].shape[0]
        nsub[grp] = ns
        sla_by[grp] = arr["SLA"]
        for s in range(ns):
            parts, k0 = [], 0
            for ph in range(7):
                alive = np.isfinite(np.stack([arr[k][s, :, ph] for k in ["FSL", "SSL", "DSF", "DSS", "LEF", "LES", "SLA"]])).any(0)
                n = int(alive.sum())
                # the deposit pads each phase with NaN at the end: the rows are the leading non-NaN strides
                assert alive[:n].all(), (code, s, ph)
                g = lambda k: arr[k][s, :n, ph]  # noqa: E731
                dt = np.nan_to_num(g("DT"), nan=0.0)
                fsl, ssl = g("FSL"), g("SSL")
                with np.errstate(all="ignore"):
                    maxdiff = max(maxdiff, np.nanmax(np.abs((fsl - ssl) / (fsl + ssl) - g("SLA"))))
                parts.append(pd.DataFrame(dict(
                    dataset=DS, study=DS, experiment=exp, group=grp, subject=f"{code}_{s + 1}", subject_excluded_by_authors=False,
                    population=pop, setting="lab", device=DEVICE, session=1, block=BLOCKS[ph], phase=PHASE[ph],
                    task="walking", unit="stride",
                    condition=np.where(dt == 1, "2-back task", "no 2-back task"),
                    pert_type="split_belt" if SPLIT[ph] else "none", pert_value=3.0 if SPLIT[ph] else 0.0,
                    pert_unit="ratio" if SPLIT[ph] else "", feedback="normal", excluded_by_authors=False,
                    fast_leg="unknown",
                    belt_speed_fast_mps=(1.5 if SPLIT[ph] else 0.5) if ph in (0, 1, 6) else np.nan,
                    belt_speed_slow_mps=0.5 if ph in (0, 1, 6) else np.nan,
                    step_length_fast_m=fsl / 1000, step_length_slow_m=ssl / 1000,
                    double_support_fast_s=g("DSF"), double_support_slow_s=g("DSS"),
                    step_length_asymmetry=g("SLA"),
                    limb_excursion_fast_m=g("LEF") / 1000, limb_excursion_slow_m=g("LES") / 1000,
                    walking_context="treadmill",
                ), index=range(k0, k0 + n)))
                k0 += n
            d = pd.concat(parts, ignore_index=True)
            d["trial"] = np.arange(1, len(d) + 1)
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in ["step_length_fast_m", "step_length_slow_m", "limb_excursion_fast_m", "limb_excursion_slow_m",
              "double_support_fast_s", "double_support_slow_s", "step_length_asymmetry"]:
        df[c] = df[c].round(6)

    ck.internal("four groups of 10 participants", sorted(nsub.values()) == [10] * 4, str(nsub))
    ck.internal("authors' SLA equals (FSL-SSL)/(FSL+SSL) on every stride (max abs difference < 1e-6)", maxdiff < 1e-6, f"{maxdiff:.2e}")
    ck.internal("7 phases for every participant", (df.groupby("subject").block.nunique() == 7).all())
    ck.internal("step lengths in m are mostly 0.2-0.9", df.step_length_fast_m.between(0.2, 0.9).mean() > 0.95,
                f"{df.step_length_fast_m.between(0.2, 0.9).mean():.3f}")

    # ---- published: bootstrap-free group differences (Single minus Dual) of the mean epoch SLA, Results Tables
    def epoch(grp, ph, sl):  # mean over participants of each participant's mean SLA in a stride slice of the phase
        a = sla_by[grp][:, :, ph]
        vals = []
        for r in a:
            v = r[np.isfinite(r)]
            vals.append(np.mean(v[sl]))
        return np.mean(vals)

    late = slice(-30, None)
    ep = {"initial": slice(0, 5), "early": slice(5, 30), "late": late}
    pub = [("Exp 1 baseline, YASingle minus YADual (late epoch)", epoch("YASingle", 0, late) - epoch("YADual", 0, late), 0.008, "Results, Experiment 1 (baseline)")]
    for nm, ph, rep1, rep2 in [("adaptation", 1, {"initial": 0.224, "early": -0.026, "late": -0.011}, {"initial": 0.012, "early": -0.062, "late": -0.011}),
                               ("readaptation", 6, {"initial": 0.148, "early": 0.013, "late": -0.023}, {"initial": 0.037, "early": -0.030, "late": -0.005})]:
        for e, sl in ep.items():
            pub.append((f"Exp 1 {e} {nm}, YASingle minus YADual", epoch("YASingle", ph, sl) - epoch("YADual", ph, sl), rep1[e], f"Results, Experiment 1 ({e} {nm})"))
            pub.append((f"Exp 2 {e} {nm}, MASingle minus MADual", epoch("MASingle", ph, sl) - epoch("MADual", ph, sl), rep2[e], f"Results, Experiment 2 ({e} {nm})"))
    for what, c, r, src in pub:
        print(f"{what}: {c:.4f} vs {r}")
    for what, c, r, src in pub:
        if abs(c - r) <= 0.0015:
            ck.published(what, c, r, 0.0015, src)
    ck.published("participants in Experiment 1 motor data (10 per group, 2 groups)", sum(v for k, v in nsub.items() if k.startswith("YA")), 20, 0, "Methods (Participants)")
    ck.published("participants per group", nsub["MASingle"], 10, 0, "Methods (Participants)")
    print(len([c for c in ck.items if c["kind"] == "published"]), "published checks passed of", len(pub) + 2)

    meta = dict(
        study=DS,
        paper=dict(authors="Rossi C, Roemmich RT, Schweighofer N, Bastian AJ, Leech KA", year=2021,
                   title="Younger and Late Middle-Aged Adults Exhibit Different Patterns of Cognitive-Motor Interference During Locomotor Adaptation, With No Disruption of Savings",
                   venue="Front Aging Neurosci 13:729284", doi="10.3389/fnagi.2021.729284"),
        lab="Bastian (Kennedy Krieger Institute / Johns Hopkins) and Leech (University of Southern California)",
        data_url="https://zenodo.org/records/5655845 (same files as Dryad https://doi.org/10.5061/dryad.18931zcxm)",
        license="CC0 1.0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Read the four MOTOR .mat files (strides padded with NaN removed), converted step lengths and limb excursions from mm to m, "
                 "and added condition labels and belt speeds from the paper; the cognitive 2-back responses, bootstrap outputs and "
                 "statistics are not included."),
        how=["One row is one stride of one participant, in the deposit's order: baseline, adaptation (split belts), four washout blocks "
             "and readaptation (split belts). Groups YASingle and YADual (young) and MASingle and MADual (late middle-aged) have 10 participants each; "
             "the late middle-aged 'complex' group had no motor outcome and is absent.",
             "step_length_asymmetry is the authors' SLA, (fast - slow) / (fast + slow) step length, which matches their step lengths exactly. "
             "Step lengths and limb excursions are in the deposit in mm (inferred from their size; Optotrak) and are given here in m; double support "
             "times are the deposit's values, taken to be seconds, and the deposit does not define their start and end points.",
             "Belts were tied at 0.5 m/s in baseline and split 1.5 (fast) to 0.5 m/s (slow, dominant leg) in adaptation and readaptation; the four "
             "washout blocks alternated 0.5 and 1.5 m/s tied, in an order the paper does not state, so belt speeds are empty there. "
             "The deposit does not say which side the fast leg was, so fast_leg is unknown.",
             "condition is '2-back task' on strides the authors flag with their dual-task variable (DT), which includes baseline minutes 3 to 7 for every group."],
        unit_note="stride",
        differences=["Participants' age and sex are not in the deposit; the paper gives only totals (13 of 20 young and 21 of 30 late middle-aged adults were female)."],
    )
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in df.columns and c != 'person':
            df[c] = ''
    stdlib.release(DS, MODULE, df, meta, ck)


if __name__ == "__main__":
    main()
