"""Werner, Struder & Donchin (2019) PLoS ONE: intermanual transfer of visuomotor adaptation and awareness.

Raw files (figshare 9780407, CC BY 4.0): s003.csv (semicolon-separated, decimal comma, 48 participants x 48 episodes of mean
movement direction, 3 header lines) and s004.csv (order of the awareness test per participant). Nothing is executed.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Werner2019_PLoSOne"
RAW = stdlib.raw_dir(DS)


def num(s):
    return float(s.replace(",", ".")) if s.strip() != "" else None


def read(fn):
    with open(RAW / fn, encoding="utf-8-sig", newline="") as f:
        return [l.split(";") for l in f.read().splitlines()]


def main():
    A = read("s003.csv")
    hdr = A[2]
    data = A[3:]
    O = [r for r in read("s004.csv")[4:]]
    ck = stdlib.Checks()
    ck.internal("48 participants x 48 episodes", len(data) == 48 and all(len(r) == 49 for r in data))
    ck.internal("awareness-order file has the same 48 rows and the same group labels in the same order",
                len(O) == 48 and [r[0] for r in O] == [r[0] for r in data])
    labels = hdr[1:]
    # phase / feedback per episode label
    spec = {"Baseline no feedback": ("baseline", "none"), "Baseline left hand": ("baseline", "normal"),
            "Baseline right hand": ("baseline", "normal"), "Adaptation": ("learning", "normal"),
            "Exclusion/inclusion": ("test", "none"), "Inclusion/exclusion": ("test", "none"),
            "Refresh": ("other", "normal"), "Intermanual transfer": ("transfer", "none"), "Deadaptation": ("washout", "normal")}
    assert set(labels) == set(spec), set(labels) - set(spec)
    rows, parts = [], []
    for i, r in enumerate(data):
        g = r[0]
        size = int(g[1:])
        gradual = g[0] == "G"
        sid = f"p{i + 1}"
        parts.append(dict(person=sid, awareness_test_order=("exclusion first" if O[i][1].strip() == "1" else "inclusion first")))
        k_ad = 0
        for j, lab in enumerate(labels):
            ph, fb = spec[lab]
            if lab == "Adaptation":
                k_ad += 1
                rot = min(3 * k_ad, size) if gradual else size
            elif lab == "Refresh":
                rot = size
            else:
                rot = 0
            rows.append(dict(dataset=DS, study="Werner 2019 PLoS ONE", experiment="intermanual-transfer", group=g, subject=sid,
                             person=sid, subject_excluded_by_authors=False, age=None, sex=None, handedness="right",
                             population="healthy", setting="lab", device="digitizing tablet and pen, view via mirror",
                             session=1, block=lab, trial=j + 1, phase=ph, time_s=None, task="reaching",
                             unit="participant summary", condition=lab,
                             pert_type="visuomotor_rotation" if rot else "none", pert_value=rot, pert_unit="deg",
                             feedback=fb, excluded_by_authors=False,
                             measure="mean_movement_direction",
                             measure_description="Mean movement direction relative to the target, 150 ms after movement onset, over one 35 s episode",
                             statistic="mean", value=num(r[1 + j]), value_unit="deg", n_trials=None, session_context=None))
    ck.internal("all 2304 episode values present", len(rows) == 2304 and all(x["value"] is not None for x in rows))
    # published checks (Results: final adaptation, last of the 25 adaptation episodes, mean +- error)
    last = labels.index("Adaptation") + 24
    assert labels[last] == "Adaptation" and labels[last + 1] != "Adaptation"
    for g, rep in (("G30", -25.5), ("S30", -24.5), ("G75", -51.5), ("S75", -70.3)):
        v = [num(r[1 + last]) for r in data if r[0] == g]
        ck.published(f"final adaptation mean direction, group {g} (deg)", sum(v) / len(v), rep, 0.05, "Results, Adaptation")
    ck.published("participants", len(data), 48, 0, "Methods, Participants")
    ck.published("adaptation episodes", labels.count("Adaptation"), 25, 0, "Methods, Procedure")
    ck.published("episodes in total", len(labels), 48, 0, "Methods, Statistics (48 episodes modelled)")
    meta = dict(
        study=DS,
        paper=dict(authors="Werner S, Struder HK, Donchin O", year=2019,
                   title="Intermanual transfer of visuomotor adaptation is related to awareness", venue="PLoS ONE 14(9):e0220748",
                   doi="10.1371/journal.pone.0220748"),
        lab="Werner (German Sport University Cologne)", data_url="https://plos.figshare.com/articles/dataset/Intermanual_transfer_of_visuomotor_adaptation_is_related_to_awareness/9780407",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Read the semicolon-separated file with decimal commas, turned it into a long table with one row per participant and "
                 "episode, labelled each episode with the authors' header, replaced participant order with codes, and moved the "
                 "awareness-test order to participants.csv."),
        how=["One row is the mean movement direction (degrees, relative to the target, measured 150 ms after movement onset) of one "
             "participant in one 35 s episode of out-and-back pen movements to 8 targets. The deposit shares episode means, not single "
             "trials; trial is the episode number from 1 and block is the authors' header for it.",
             "Groups: G or S for gradual (3 deg per episode) or sudden rotation, then 30 or 75 deg. Negative values are movements "
             "opposite to the rotation (adaptation is negative in the deposit); signs are kept as shared.",
             "Inferred by us, not stated in the deposit: pert_value of each adaptation episode (gradual groups rise 3 deg per episode up "
             "to the final size, sudden groups are at full size from episode 1), feedback for the baseline left- and right-hand "
             "episodes (normal), and that s004 rows follow the same participant order as s003 (the two files give no ID). The "
             "authors' headers for the awareness test are 'Exclusion/inclusion' then 'Inclusion/exclusion'; "
             "awareness_test_order in participants.csv is the flag from s004 (1 = exclusion first). Age and sex are not in the deposit."],
        unit_note="mean movement direction of one participant in one episode",
        differences=["The paper does not say whether the printed +/- values of final adaptation are SD or SE; only the means are checked."],
    )
    pc = [("awareness_test_order", "category", "exclusion first, inclusion first",
           "Which awareness test came first, from s004 (paired with s003 by row position).")]
    stdlib.release(DS, "summary", rows, meta, ck, participants=parts, participant_columns=pc)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
