"""Weekly finder, step 6: the weekly report (Markdown, posted as a GitHub issue by the workflow).

usage: python3 -I tools/finder/report.py [--std work/std] [--out work/finder/report.md] [--run-url URL]

Built only from the run's own outputs (work/finder/*.json[l] and work/std/<id>/result.json), so it can be written even when
an earlier job failed. It holds public metadata only (titles, DOIs, links, decisions), never data values."""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SITE = "https://physicalintelligencelab.github.io/openmotor/"


def md(s, n=200):
    """Plain, short, safe Markdown: no table breaks, no @mentions, no HTML."""
    s = C.clean_text(str(s or ""))
    s = s.replace("|", "/").replace("@", "(at)").replace("<", "(").replace(">", ")").replace("`", "'")
    return s if len(s) <= n else s[:n - 1] + "..."


def link(text, url):
    url = str(url or "")
    if not re.match(r"https://[^\s()<>]+$", url):
        return md(text)
    return f"[{md(text, 80)}]({url})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std", default=str(C.HERE / "work/std"))
    ap.add_argument("--out", default="")
    ap.add_argument("--run-url", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--work", default=str(C.WORK), help="folder with the run's finder outputs")
    a = ap.parse_args()
    W = Path(a.work)
    fs = C.load_json(W / "find_summary.json", {})
    ss = C.load_json(W / "screen_summary.json", {})
    screened = C.read_jsonl(W / "screened.jsonl")
    entries = C.load_json(W / "new_entries.json", [])
    lprob = C.load_json(W / "listing_problems.json", [])
    pub = C.load_json(W / "publish_result.json", {})
    held = C.read_jsonl(W / "held.jsonl")
    results = [C.load_json(p, {}) for p in sorted(Path(a.std).glob("*/result.json"))] if Path(a.std).exists() else []
    outcome = {o["id"]: o for o in pub.get("outcome", [])}
    have = {r.get("id") for r in results}
    results += [dict(o) for o in pub.get("outcome", []) if o["id"] not in have]  # e.g. jobs that left no result
    today = dt.date.today().isoformat()
    L = [f"Weekly finder run of {today}" + (" (dry run: nothing was published)" if a.dry_run or pub.get("dry_run") else "") + ".", ""]
    if not fs:
        L += ["**The find step did not finish**, so nothing was found, listed or standardized this week. "
              + (f"See the [workflow run]({a.run_url})." if a.run_url else ""), ""]
    else:
        dec = ss.get("decisions", {})
        rel = [r for r in results if (outcome.get(r["id"]) or r).get("status") == "released"]
        L += ["## Summary", "",
              f"- Window: {fs.get('since')} to {fs.get('until')}",
              f"- Found: {fs.get('hits', 0)} on-topic hits, {fs.get('with_paper', 0)} with a paper, "
              f"{fs.get('candidates', 0)} new candidates after removing what is listed or was seen before "
              f"({', '.join(f'{k} {v}' for k, v in (fs.get('dropped') or {}).items() if v)})",
              f"- Screened: {ss.get('screened', 0)} ({dec.get('yes', 0)} in, {dec.get('no', 0)} out, {dec.get('unclear', 0)} unclear, "
              f"{dec.get('no answer', 0)} unanswered; {ss.get('escalated', 0)} sent to the larger model; ${ss.get('cost_usd', 0)})",
              f"- Listed on the site: {len(entries)}" + (" (not pushed)" if fs and not pub.get("pushed") and not pub.get("dry_run") else ""),
              f"- Standardized: {len(rel)} of {len(results)} attempted",
              f"- Data deposits held back (no paper yet): {fs.get('held_back_new', 0)} new, {fs.get('held_total', 0)} waiting", ""]
        L += ["| source | hits | on topic | requests | seconds |", "|---|---|---|---|---|"]
        for k, v in (fs.get("per_source") or {}).items():
            L.append(f"| {k} | {v.get('hits', 0)} | {v.get('on_topic', 0)} | {v.get('requests', 0)} | {v.get('seconds', 0)}"
                     + (f" (error: {md(v['error'], 80)})" if v.get("error") else "") + " |")
        L.append("")
    if entries:
        why = {r["doi"]: r for r in screened}
        L += ["## Listed this week", "", "| study | task | data | why it is in |", "|---|---|---|---|"]
        for e in entries:
            p = e["papers"][0]
            r = why.get(p["doi"]) or next((x for x in screened if x.get("cid") == (e.get("found_by") or {}).get("cid")), {})
            L.append(f"| {link(e['label'] + ' (' + e['id'] + ')', SITE + '#study/' + e['id'])}: {md(p['title'], 110)} "
                     f"({link('doi', 'https://doi.org/' + p['doi'])}) | {md(e['task'])} | {link(e['host'], e['data_url'])} | "
                     f"{md(((r.get('screen') or {}).get('reason')), 160)} |")
        L.append("")
    if results:
        L += ["## Standardization", "", "| study | result | details |", "|---|---|---|"]
        for r in results:
            o = outcome.get(r["id"]) or r
            st = o.get("status")
            if st == "released":
                det = f"{r.get('module')}, {r.get('rows')} rows, {r.get('persons')} participants, {r.get('published_checks')} checks against the paper"
                if o.get("url", "").startswith("https://"):
                    det += f" ({link('ZIP', o['url'])})"
            elif st == "skipped":
                det = md(r.get("reason"), 250)
            else:
                det = md(o.get("reason") or r.get("reason"), 250) + (" Recorded as skipped after the last attempt." if o.get("final") else
                                                                    " Will be retried next week.")
            L.append(f"| {r['id']} | {st} | {det} |")
        L.append("")
    needs = []
    for r in screened:
        s = r.get("screen") or {}
        if s.get("include") == "unclear":
            needs.append(f"Unclear after two screens: {link(r.get('title', '')[:90], 'https://doi.org/' + r['doi'])} "
                         f"(data: {link(r.get('host') or 'link', r.get('data_url'))}): {md(s.get('reason'), 200)}")
    for p in lprob:
        needs.append(f"Included but not listed: {md(p.get('title'), 100)} (doi {md(p.get('doi'), 60)}): {md(p.get('problem'), 150)}")
    for r in results:
        o = outcome.get(r["id"]) or r
        np_ = (r.get("agent") or {}).get("needs_person")
        if np_:
            needs.append(f"{r['id']}: {md(np_, 250)}")
        for w in (r.get("gates") or {}).get("warnings", [])[:3]:
            if "tolerance" in w and o.get("status") == "released":
                needs.append(f"{r['id']}: {md(w, 200)}")
        if o.get("final"):
            needs.append(f"{r['id']}: gave up after the automatic attempts ({md(o.get('reason'), 200)})")
    for n in (fs.get("notes") or []):
        needs.append(md(n, 200))
    errs = [f"{md(host, 60)} {n}" for host, n in (fs.get("http_errors") or {}).items() if n >= 10]
    if errs:
        needs.append("Many failed requests (results may be incomplete; check the find log): " + ", ".join(errs))
    if ss.get("problems"):
        needs.append(f"Screening problems: {md('; '.join(ss['problems'][:3]), 250)}")
    if fs and not pub:
        needs.append("The publish job did not run or did not finish: nothing was committed this week.")
    elif pub.get("error"):
        needs.append(f"Publishing failed: {md(pub['error'], 200)}")
    L += ["## Needs a person", ""] + ([f"- {n}" for n in needs] if needs else ["Nothing this week."]) + [""]
    outs = [r for r in screened if (r.get("screen") or {}).get("include") == "no"]
    if outs:
        L += ["<details><summary>Screened out (" + str(len(outs)) + ")</summary>", ""]
        for r in outs[:60]:
            L.append(f"- {link(r.get('title', '')[:100], 'https://doi.org/' + r['doi'])}: {md((r.get('screen') or {}).get('reason'), 160)}")
        L += ["", "</details>", ""]
    if held:
        L += ["<details><summary>Data deposits without a paper, held back (" + str(len(held)) + " new)</summary>", ""]
        for x in held[:25]:
            L.append(f"- {link(x.get('title', '')[:100] or x.get('data_url'), x.get('data_url'))} ({md(x.get('host'), 30)})")
        L += ["", "</details>", ""]
    L += ["---", "Made by the weekly finder (.github/workflows/finder.yml; tools/finder/README.md says how it works and how to "
          "turn it off)." + (f" [Workflow run]({a.run_url})." if a.run_url.startswith("https://") else "")]
    text = "\n".join(L)
    if len(text) > 60000:
        text = text[:59000] + "\n\n(report cut at 60,000 characters)\n"
    out = Path(a.out) if a.out else W / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"report: {len(text)} characters")


if __name__ == "__main__":
    main()
