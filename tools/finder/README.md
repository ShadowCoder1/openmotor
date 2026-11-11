# Weekly finder

Finds newly shared human motor control and motor learning datasets, lists them on the site, standardizes up to six a
week, checks them and publishes them, with no person in the loop. It runs as `.github/workflows/finder.yml` every
Saturday night (US Eastern) and posts a "Weekly finder report YYYY-MM-DD" issue.

## How a run flows
1. **find** (`find.py`, no AI): searches Zenodo, Figshare, OSF (projects and preprints with data links), DataCite
   (datasets, and datasets that name a paper DOI), Europe PMC (papers and preprints; repository links from the data
   availability statements of open full texts and bioRxiv JATS) and OpenAlex (new works by the last authors of listed
   studies; works citing 16 key papers listed in `terms.py`) for the phrases in `terms.py`, since the last run (minus a
   7-day overlap). Papers with no data link yet are matched by title against DataCite and OSF deposits (authors often
   name the deposit after the paper) and kept on a recheck list for 90 days, because open full texts reach Europe PMC
   weeks after publication. A candidate needs a paper DOI and a public data location; deposits without a paper are held
   back and rechecked for up to 180 days. Anything already listed (paper DOI, normalized data link, or title) or seen
   before is dropped. Each source has a time cap (`SOURCE_MINUTES` in find.py). Output: `work/finder/candidates.jsonl`.
2. **screen** (`screen.py`): Claude Haiku screens batches of 15 (human? motor behaviour? the authors' own deposit?
   trial- or participant-level data?), Sonnet takes a second look at the unclear ones. No tools; candidate text is passed
   as data; answers are validated.
3. **listing** (`listing.py`): builds `source/extra_studies.json` entries (Crossref, OpenAlex, the repository API) with
   ids like `Smith2026_JNeurophysiol`. Included studies are listed whether or not standardization works.
4. **standardize** (one job per study, at most 3 at once, 75 minutes each): a Claude Code agent follows `BRIEF.md`
   (sandboxed, network limited to repositories and paper sources, settings in `agent_settings.json`); then
   `standardize.py gates`, run from a copy of `tools/` made before the agent started, checks the checkout against a
   manifest, the ZIP and its record (a passing check against a number in the paper is required) and that task modules
   are plain data; then an independent reviewer follows `REVIEW.md`. Only a "publish" verdict lets the ZIP through.
5. **publish** (`publish.py`): uploads ZIPs to the `standardized` release, then commits entries, records, skip records,
   converters, modules, abstracts and `source/finder/state.json` on the newest main (retrying if someone pushed first)
   and runs `tools/explore_build.py --incremental` if it exists. Failed studies are retried once more the next week, then
   recorded as skipped. Then `pages.yml` rebuilds the site and `report.py` writes the issue.

## Running it by hand
- In GitHub: Actions > Weekly finder > Run workflow (inputs: how many to standardize, dry run, start date). A dry run
  does everything except publishing; its report goes to the run summary instead of an issue.
- Locally (small tests; uses this machine's Claude login for the screen):
  ```
  python3 -I tools/finder/find.py --since 2026-10-01 --state /tmp/finder/state.json --authors /tmp/finder/authors.json --out work/finder
  python3 -I tools/finder/screen.py --limit 20
  python3 -I tools/finder/listing.py --limit 3
  python3 -I tools/finder/report.py
  ```
  Without `--state`, find.py reads `source/finder/state.json`; it never writes there (publish.py does).

## Where state lives
- `source/finder/state.json`: `last_run`, `seen` (DOIs and data keys already screened), `held` (deposits waiting for a
  paper), `watch` (papers waiting for a data link), `pending` (listed studies waiting for standardization, with attempts
  and the last problem), `history`. It is created by the first run; without it the first run searches the last 14 days
  (start further back with the `since` input).
- `source/finder/authors.json`: cached OpenAlex ids of listed studies' last authors and of the key papers.
- Each run's working files are kept as workflow artifacts for 7 days (`finder-run`, `std-<id>`, `finder-published`).

## Turning it off
- Pause: Actions > Weekly finder > "..." > Disable workflow (or `gh workflow disable finder.yml`).
- List only, no standardizing: set the default of `max_standardize` to "0" in `.github/workflows/finder.yml`.
- Undo a listing: remove its entry from `source/extra_studies.json` (and its record); keep its DOI in `state.json`
  `seen` so it is not found again.

Secrets: `CLAUDE_CODE_OAUTH_TOKEN` (required; AI steps only) and `OPENALEX_API_KEY` (optional, free at
https://openalex.org/settings/api; without it OpenAlex shares a small daily budget per IP address).
