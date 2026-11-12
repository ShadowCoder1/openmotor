"""Plausibility checks on a standardized reaching table (reach90) that its own format checks cannot see.

usage: python3 -I tools/std/sanity.py <dataset ZIP or CSV> [...]    prints problems; exit status 1 when there are any

At every point where the rotation changes within a sequence (rotation known on both sides), hand_angle is compared
with the mean of the 5 reaches before the change, in units of the change and signed so that compensating it is positive
(c = 0: no change; c = 1: full compensation). Group means per experiment and change are used, not single people;
changes under 10 deg or shared by fewer than 3 sequences are skipped (too noisy).

  1. moves with the rotation at once: c on the first two reaches <= -0.4. A hand cannot follow a rotation the
     participant has not seen yet; a cursor does at once. The column holds the cursor (cursor = hand + rotation).
  2. drifts the wrong way: c over the last 8 reaches of the new rotation <= -0.3. People compensate against a
     rotation, so a sign is reversed (in hand_angle or in rotation).
  3. full compensation at once that then fades: c on the first two reaches >= 0.6 and the late c under half of it,
     while nobody was told about the rotation or given an aim. This is cursor data with a reversed rotation sign
     (Chu & Sanger 2014, version 1).

Found 2026-11-12: Chu2014_F1000Res version 1 put the authors' cursor directions in hand_angle with the paper's rotation
signs, which are the reverse of the files' frame. An audit of every rotation curve in the Explore data (278 lines, 83
datasets) found no other case.
"""
import csv
import io
import statistics as st
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

PRE, EARLY, LATE, MIN_NEW, MIN_CHANGE, MIN_SEQ = 5, 2, 8, 10, 10, 3


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".csv") and not n.endswith("columns.csv")]
            main = [n for n in names if Path(n).stem == Path(n).parent.name] or names
            return list(csv.DictReader(io.StringIO(z.read(main[0]).decode("utf-8-sig"))))
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def rotation_changes(rows):
    """[(experiment, change number, delta, c_early, c_late, n sequences, told)] from group means."""
    if not rows or "rotation" not in rows[0] or "hand_angle" not in rows[0]:
        return []
    seqs = defaultdict(list)
    for r in rows:
        seqs[r.get("subject", "")].append(r)
    acc = defaultdict(lambda: {"e": [], "l": [], "told": False})
    for subj, rs in seqs.items():
        rs.sort(key=lambda r: _num(r.get("trial")) or 0)
        rot = [_num(r.get("rotation")) for r in rs]
        hand = [_num(r.get("hand_angle")) if (r.get("trial_kind") or "reach") == "reach" else None for r in rs]
        k = 0
        for i in range(1, len(rs)):
            if rot[i] is None or rot[i - 1] is None or rot[i] == rot[i - 1]:
                continue
            k += 1
            j = i
            while j < len(rs) and rot[j] == rot[i]:
                j += 1
            pre = [h for h in hand[max(0, i - 3 * PRE):i] if h is not None][-PRE:]
            new = [h for h in hand[i:j] if h is not None]
            if len(pre) < PRE or len(new) < MIN_NEW:
                continue
            d = rot[i] - rot[i - 1]
            if abs(d) < MIN_CHANGE or abs(d) > 180:
                continue
            base = st.mean(pre)
            c = lambda x: (x - base) * (-1 if d > 0 else 1) / abs(d)  # noqa: E731
            key = (rs[i].get("experiment", ""), k, d)
            acc[key]["e"].append(c(st.mean(new[:EARLY])))
            acc[key]["l"].append(c(st.mean(new[-LATE:])))
            told = (rs[i].get("instr_strategy") == "given_with_angle"
                    or rs[i].get("instr_disclosure") in ("exists", "exact"))
            acc[key]["told"] = acc[key]["told"] or told
    out = []
    for (exp, k, d), a in sorted(acc.items()):
        if len(a["e"]) >= MIN_SEQ:
            out.append((exp, k, d, st.mean(a["e"]), st.mean(a["l"]), len(a["e"]), a["told"]))
    return out


def problems(rows):
    p = []
    for exp, k, d, ce, cl, n, told in rotation_changes(rows):
        where = f"{exp}, rotation change {k} ({d:+g} deg, {n} sequences)"
        if ce <= -0.4:
            p.append(f"{where}: hand_angle moves with the rotation on the first reaches (c = {ce:.2f}); "
                     "it looks like the cursor direction (cursor_error), not the hand")
        if cl <= -0.3:
            p.append(f"{where}: hand_angle drifts with the rotation, not against it (late c = {cl:.2f}); "
                     "a sign is reversed in hand_angle or rotation")
        if ce >= 0.6 and cl < ce / 2 and not told:
            p.append(f"{where}: full compensation on the first reaches that then fades (c = {ce:.2f} -> {cl:.2f}); "
                     "it looks like cursor data with a reversed rotation sign")
    return p


def main(paths):
    bad = 0
    for path in paths:
        ps = problems(read_rows(path))
        bad += bool(ps)
        print(f"{Path(path).name}: " + ("ok" if not ps else f"{len(ps)} problem(s)"))
        for x in ps:
            print("  - " + x)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
