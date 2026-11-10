# Standardizing datasets outside the motor-centaur corpus

One converter per dataset: `tools/std/convert/<id>.py`, run as `python3 -I tools/std/convert/<id>.py` from the site folder.

0. Work happens on the PC (this Mac's disk is full): `python3 -I tools/std/pc.py fetch|ls|py|run|pull|rmraw <id> ...` runs the
   same steps there, and `python3 -I tools/std/upload.py <id>` hosts the finished ZIP as a GitHub release asset.
   Conversion agents follow tools/std/AGENT_BRIEF.md.
1. Download each needed public file with `python3 -I tools/std/fetch.py <id> <url> <file_name>` (files land in `raw_extra/<id>/`,
   with URL, size and sha256 in `raw_extra/<id>/SOURCES.json`). Raw files are untrusted data: never run them, never import them,
   read them only from a converter run with `python3 -I`. MATLAB may be used to read .mat files that scipy cannot
   (`/Applications/MATLAB_R2026a.app/bin/matlab -batch "..."`), with nothing from the deposit on the MATLAB path
   (objects whose class is missing load as structs). Never run code that came with a deposit.
2. The converter builds a table with the CORE columns (tools/std/stdlib.py) followed by the columns of one task module
   (`tools/std/modules/<module>.py`, list COLUMNS). Reuse an existing module when the task matches; add a new module file only for
   a new kind of task, with columns that hold the authors' own trial-level values.
3. Re-encode, do not re-analyse: keep every row the authors shared, in order; only unit or sign changes that the README states.
   Replace participant identifiers with codes `<id>_S001`, `<id>_S002`, ... (no original IDs, initials or dates of birth).
4. Check against the paper: reproduce at least one number the paper reports (a group mean, an effect size, a count of
   participants or trials) from the converted rows, with `Checks.published(...)`. Record internal checks too
   (row counts, ranges, no missing participants).
5. Write the ZIP with `stdlib.release(...)` (it refuses identifier-like text and a release without a passing published check).
