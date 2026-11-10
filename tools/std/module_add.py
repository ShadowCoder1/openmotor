"""Append a column to an existing task module, safely while other agents run (file lock; existing columns never change).

usage: python3 -I tools/std/module_add.py <module> <column> <type> "<values or units>" "<description>"

The column goes at the end of COLUMNS in tools/std/modules/<module>.py. Converters written earlier keep working: release()
writes a module column a row leaves out as empty. Use this instead of editing a module file you did not create."""
import fcntl
import re
import sys
from pathlib import Path

MOD = Path(__file__).resolve().parent / "modules"


def main(module, column, ctype, values, description):
    assert re.fullmatch(r"[a-z][a-z0-9_]*", column), "column names are lower_snake_case"
    assert ctype in ("text", "number", "integer", "category", "true/false"), ctype
    p = MOD / f"{module}.py"
    assert p.exists(), f"no module {module}; create tools/std/modules/{module}.py instead"
    with open(MOD / ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        s = p.read_text()
        ns0 = {}
        exec(compile(s, str(p), "exec"), ns0)  # our own module file
        if any(c[0] == column for c in ns0["COLUMNS"]):
            print(f"{module}: column {column} already exists")
            return 0
        i = s.rindex("]")
        entry = f"    ({column!r}, {ctype!r}, {values!r}, {description!r}),\n"
        s = s[:i].rstrip() + "\n" + entry + s[i:]
        s = s.replace("',\n    (", "',\n    (")
        p.write_text(s)
        ns = {}
        exec(compile(p.read_text(), str(p), "exec"), ns)  # our own module file: check it still parses and holds the column
        assert any(c[0] == column for c in ns["COLUMNS"])
    print(f"{module}: added column {column}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
