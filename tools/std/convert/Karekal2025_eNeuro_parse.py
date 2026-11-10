"""Reader for the behavioural CSV files of Karekal et al. (2025), eNeuro (OSF 8vqu6, Behavioral_Data_All_Groups).

The files went through a spreadsheet before they were shared and exist in three layouts: (1) 27 or 28 columns with the six trace lists in
single cells (lists separated by spaces or tabs); (2) the same table in which every list value sits in its own cell, so that a row has
hundreds of cells, and the comma that separated two lists ended up inside one cell holding the last value of one list and the first of
the next ("a,b"; when the spreadsheet read such a cell as a number the digits were joined and printed with thousands separators, e.g.
1,193,337 for 1193 and 337); (3) rows without any trace. The reader rebuilds the six lists (go_pos_x, go_pos_y, go_times, stop_pos_x,
stop_pos_y, stop_times) of layout 2 from the cell count, which fixes the list lengths, and splits each joined cell where the neighbouring
values continue best. A row it cannot rebuild is returned with status 'unparsed' and no lists. Nothing in the files is executed."""
import csv, re
import numpy as np


def candidates(cell, prev, nxt, kind=None):
    """Boundary cell = last value of one list and first value of the next. The authors' spreadsheet export joined them in one cell, as text
    (420,2) or, when it read the cell as a number, with thousands separators (1,193,337). Try every split of the digits and keep the one
    that continues the neighbouring values best."""
    parts = cell.split(",")
    if not all(p.isdigit() for p in parts):
        return None
    d = "".join(parts)
    best = None
    commas = {len(parts[0])} if len(parts) == 2 else set()  # where the printed comma is (text cells keep their comma)
    only = len(parts) == 2 and len(parts[1]) != 3  # a,b with b not three digits was never read as a number
    for i in range(1, len(d)):
        if only and i not in commas:
            continue
        if d[i] == "0" and len(d) - i > 1:
            continue
        L, R = int(d[:i]), int(d[i:])
        c = (abs(L - prev) if prev is not None else 0) + (abs(R - nxt) if nxt is not None else 0)
        c += 0 if i in commas else 0.5
        if kind is not None:
            lim = {"x": (0, 2000), "y": (0, 1200), "t": (0, 120000)}
            lo_l, hi_l = lim[kind[0]]
            lo_r, hi_r = lim[kind[1]]
            if kind[1] == "t":
                hi_r = 4000  # first time stamp of a list
            if not (lo_l <= L <= hi_l and lo_r <= R <= hi_r):
                c += 10 ** 7
        if best is None or c < best[0]:
            best = (c, L, R)
    return best


def scalars(tokens, wide=True):
    """pid, [age], gender, block, trial, fix_time, signal, count, stop_time, then goRT and RT (one of them may be missing in some rows)"""
    i = 1
    age = ""
    if tokens[i] == "":
        i += 1
    elif not (tokens[i].isalpha() and tokens[i] != "XX"):
        age = tokens[i]
        i += 1
    d = dict(pid=tokens[0], age=age, gender=tokens[i], block=tokens[i + 1], trial=tokens[i + 2], fix_time=tokens[i + 3],
             signal=tokens[i + 4], count=tokens[i + 5], stop_time=tokens[i + 6])
    rest = tokens[i + 7:]
    if wide:
        assert len(rest) == 2, rest
        d["goRT"], d["RT"] = rest
        return d, i + 9
    if len(rest) >= 4 and rest[-3:] == ["", "", ""]:
        d["go_lists_empty"] = True  # the three empty fields before the first value are the go lists
        rest = rest[:-3]
    if len(rest) == 2:
        d["goRT"], d["RT"] = rest
    else:
        assert len(rest) == 1, rest
        d["goRT"], d["RT"] = "?", rest[0]  # one value only: which of the two it is must be settled from the tail
        d["one_value"] = True
    return d, len(tokens)


def scalars_single(c0):
    """a row with no trace: scalars then six empty fields, then exclude and the fixed fields"""
    j = next(i for i in range(8, len(c0)) if c0[i] == "")  # first empty list field
    d, k = scalars(c0[:j], False)
    return d, j


def meta_split(rest):
    return ",".join(rest[:-10]), rest[-10:]


def ints(s):
    return [int(v) for v in s.split()] if s.strip() else []


def parse_wide(r):
    """27 or 28 cells: scalars (age may be missing), six space- or tab-separated lists, exclude, focus ... window"""
    sc, k = scalars(r[:11], True) if len(r) == 28 else scalars(r[:10], True)
    lists = tuple(ints(c) for c in r[k:k + 6])
    assert len(r) - k == 17, (len(r), k)
    return dict(sc, exclude=r[k + 6], fixed=r[k + 7:k + 17], lists=lists, status="ok")


def parse_row(r):
    d = _parse_row(r)
    m = re.fullmatch(r"(\d+),(\d+),(\d+),(.*)", d.get("exclude", ""), flags=re.S)
    if m and d.get("lists") is not None and not d["lists"][3]:
        # a stop segment of ONE sample (x, y, time) sits between the last go value and the exclude text; it was read as part of the text
        sx, sy, st = int(m.group(1)), int(m.group(2)), int(m.group(3))
        gx, gy, gt = d["lists"][:3]
        d = dict(d, exclude=m.group(4), lists=(gx, gy, gt, [sx], [sy], [st]), status="ok")
    return d


def _parse_row(r):
    while r and r[-1] == "":
        r = r[:-1]
    c0 = r[0].split(",")
    if len(r) == 1:
        sc, k = scalars_single(c0)
        assert all(t == "" for t in c0[k:k + 6]), c0
        exclude, fixed = meta_split(c0[k + 6:])
        return dict(sc, exclude=exclude, fixed=fixed, lists=None, status="no trace")
    sc, k = scalars(c0[:-1], False)
    first = int(c0[-1])
    N = len(r)
    tail = r[-1].split(",")
    last_value = int(tail[0])
    cells = r[1:-1]
    inner_q = {i for i, c in enumerate(cells, start=1) if "," in c}
    if tail[1:4] == ["", "", ""]:
        exclude, fixed = meta_split(tail[4:])
        hyps = [(N - 1) // 3]
        assert (N - 1) % 3 == 0
        a = hyps[0]
        bset = {a, 2 * a}
        res = try_hyp(first, cells, last_value, a, None, bset, inner_q)
        if res is None:
            return dict(sc, exclude=exclude, fixed=fixed, lists=None, status="unparsed: go only")
        return dict(sc, exclude=exclude, fixed=fixed, lists=res[1], status="go only")
    exclude, fixed = meta_split(tail[1:])
    if sc.get("go_lists_empty"):
        a = (N - 1) // 3
        res = try_hyp(first, cells, last_value, a, None, {a, 2 * a}, inner_q) if (N - 1) % 3 == 0 else None
        if res is None:
            return dict(sc, exclude=exclude, fixed=fixed, lists=None, status="unparsed: stop lists only")
        gx, gy, gt = res[1][:3]
        return dict(sc, exclude=exclude, fixed=fixed, lists=([], [], [], gx, gy, gt), status="stop only")
    if (N - 1) % 3 != 0:
        return dict(sc, exclude=exclude, fixed=fixed, lists=None, status=f"unparsed: cells {N}")
    K = (N - 1) // 3
    found = []
    # go-only variant without commas for the empty stop lists
    if inner_q <= {K, 2 * K}:
        res = try_hyp(first, cells, last_value, K, None, {K, 2 * K}, inner_q)
        if res is not None:
            found.append((res[0], res[1], "go only"))
    for a in range(1, K):
        b = K - a
        bset = {a, 2 * a, 3 * a, 3 * a + b, 3 * a + 2 * b}
        if not inner_q <= bset:
            continue
        res = try_hyp(first, cells, last_value, a, b, bset, inner_q)
        if res is not None:
            found.append((res[0], res[1], "ok"))
    if not found:
        return dict(sc, exclude=exclude, fixed=fixed, lists=None, status="unparsed: no consistent split")
    found.sort(key=lambda f: f[0])
    if len(found) > 1 and found[1][0] < 3 * found[0][0] + 20:
        return dict(sc, exclude=exclude, fixed=fixed, lists=None, status="unparsed: ambiguous")
    return dict(sc, exclude=exclude, fixed=fixed, lists=found[0][1], status=found[0][2])


KINDS = {("a", 1): ("x", "y"), ("a", 2): ("y", "t"), ("a", 3): ("t", "x"), ("b", 1): ("x", "y"), ("b", 2): ("y", "t")}
KINDS_NOSTOP = {1: ("x", "y"), 2: ("y", "t")}


def try_hyp(first, cells, last_value, a, b, bset, inner_q):
    """a = length of a go list - 1, b = length of a stop list - 1 (None: no stop lists). Return (cost, lists) or None."""
    plain = {}
    for i, c in enumerate(cells, start=1):
        if i not in bset:
            if not c.isdigit():
                return None
            plain[i] = int(c)
    vals, cost = [], 0
    for i, c in enumerate(cells, start=1):
        if i in bset:
            prev = plain.get(i - 1, first if i == 1 else None)
            nxt = plain.get(i + 1)
            kind = KINDS_NOSTOP.get(i // a if i % a == 0 else None) if b is None else KINDS.get(("a", i // a) if i % a == 0 and i <= 3 * a else ("b", (i - 3 * a) // b if b and (i - 3 * a) % b == 0 else None))
            cand = candidates(c, prev, nxt, kind)
            if cand is None:
                return None
            cost += cand[0]
            vals += [cand[1], cand[2]]
        else:
            vals.append(plain[i])
    seq = [first] + vals + [last_value]
    n = a + 1
    if b is None:
        if len(seq) != 3 * n:
            return None
        gx, gy, gt = seq[:n], seq[n:2 * n], seq[2 * n:]
        lists = (gx, gy, gt, [], [], [])
    else:
        m = b + 1
        if len(seq) != 3 * n + 3 * m:
            return None
        gx, gy, gt = seq[:n], seq[n:2 * n], seq[2 * n:3 * n]
        sx, sy, st = seq[3 * n:3 * n + m], seq[3 * n + m:3 * n + 2 * m], seq[3 * n + 2 * m:]
        lists = (gx, gy, gt, sx, sy, st)
    for t in (lists[2], lists[5]):
        if len(t) > 1 and np.any(np.diff(t) < 0):
            return None
    for xs in (lists[0], lists[3]):
        if len(xs) and (min(xs) < 0 or max(xs) > 2000):
            return None
    for ys in (lists[1], lists[4]):
        if len(ys) and (min(ys) < 0 or max(ys) > 1200):
            return None
    return cost, lists


def read_file(path):
    rows = list(csv.reader(open(path, newline="", encoding="utf-8", errors="replace")))[1:]
    out = []
    for r in rows:
        if not any(r):
            continue
        if len(r) in (27, 28) and not any("," in c for c in r):
            out.append(parse_wide(r))
        else:
            try:
                out.append(parse_row(r))
            except Exception as e:
                try:
                    out.append(fallback(list(r)))
                except Exception:
                    out.append(dict(status="unparsed: row not read", pid=r[0][:25], lists=None))
    return out


def fallback(r):
    """Scalars and exclusion text of a row whose traces could not be rebuilt."""
    while r and r[-1] == "":
        r = r[:-1]
    c0 = r[0].split(",")
    i = 1
    age = ""
    if c0[i] == "":
        i += 1
    elif not (c0[i].isalpha() and c0[i] != "XX"):
        age = c0[i]
        i += 1
    d = dict(pid=c0[0], age=age, gender=c0[i], block=c0[i + 1], trial=c0[i + 2], fix_time=c0[i + 3], signal=c0[i + 4], count=c0[i + 5],
             stop_time=c0[i + 6], goRT="?", RT="?")
    tail = r[-1].split(",")
    d.update(exclude=tail[-11], fixed=tail[-10:], lists=None, status="unparsed: traces not rebuilt")
    return d
