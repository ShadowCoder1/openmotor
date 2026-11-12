"""Weekly finder, step 5 (deterministic): publish this run's results.

usage: python3 -I tools/finder/publish.py [--std work/std] [--dry-run]

Reads the find job's outputs (work/finder/: new_entries.json, new_abstracts.json, state.json) and one folder per
standardized study (work/std/<id>/: result.json and files/ as written by standardize.py collect). For every study the
reviewer passed, it checks the ZIP again, uploads it to the release 'standardized' with gh (GH_TOKEN), and records its URL.
Then it commits the new listing entries, records, skip records, converters, task modules, abstracts and the finder state
on top of the newest main, retrying when another run pushed first (like tools/intake/worker.py commit_files). Studies that
failed are retried on a later run (up to MAX_ATTEMPTS), then recorded as skipped. Writes work/finder/publish_result.json
and the GitHub output changed=yes|no."""
import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import standardize as S  # noqa: E402

REPO = "physicalintelligencelab/openmotor"
TAG = "standardized"
HERE = C.HERE


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=HERE, check=check, capture_output=True, text=True)


def gh(*args, check=True):
    return subprocess.run(["gh", *args], cwd=HERE, check=check, capture_output=True, text=True)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "std"))
from versioning import publish_zip  # noqa: E402


def upload(ds, zpath, rec):
    """Publish as a new version (never overwrites); updates rec. Returns publish_zip's result."""
    for attempt in range(3):
        try:
            return publish_zip(ds, zpath, TAG, repo=REPO, record=rec)
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(20 * (attempt + 1))


def verify_release(ds, files_dir):
    """The publish job's own check of what a standardize job sent (it never trusts the artifact blindly)."""
    rec_p = files_dir / "source/release_extra_records" / f"{ds}.json"
    zpath = files_dir / "source/release_extra" / f"{ds}.zip"
    if not rec_p.exists() or not zpath.exists():
        return ["the ZIP or its record is missing from the artifact"], None
    rec = json.load(open(rec_p))
    problems = []
    if rec.get("id") != ds:
        problems.append("record id differs")
    if not rec.get("checks") or not all(c.get("ok") for c in rec["checks"]):
        problems.append("a check failed")
    if rec.get("no_published_reason") or not any(c.get("kind") == "published" and c.get("ok") for c in rec["checks"]):
        problems.append("no passing published check")
    for m in (files_dir / "tools/std/modules").glob("*.py") if (files_dir / "tools/std/modules").exists() else []:
        try:
            S.module_columns(m.read_text(encoding="utf-8"))
        except (ValueError, SyntaxError) as e:
            problems.append(f"module {m.stem} is not plain data: {e}")
    for f in files_dir.rglob("*"):
        if f.is_file() and not S.allowed(str(f.relative_to(files_dir)), ds):
            problems.append(f"unexpected file {f.relative_to(files_dir)}")
    try:
        mod = rec.get("module") or ""
        mdir = files_dir / "tools/std/modules" if (files_dir / "tools/std/modules" / f"{mod}.py").exists() else HERE / "tools/std/modules"
        cols, stdlib = S.expected_columns(mod, mdir, HERE / "tools/std", HERE / "source/columns.csv")
    except Exception as e:  # noqa: BLE001
        return problems + [f"module {rec.get('module')!r} unreadable: {type(e).__name__}"], rec
    zp, _ = S.zip_problems(zpath, ds, rec, cols, stdlib)
    return problems + zp + [f"plausibility: {x}" for x in S.plausibility(zpath)], rec


def merge_module(main_text, ours_text):
    """Our module file, re-applied on the newest main: new modules as they are; for an existing module, append the columns
    we added that main does not have yet (both sides only ever append)."""
    if main_text is None:
        return ours_text
    _, mcols = S.module_columns(main_text)
    _, ocols = S.module_columns(ours_text)
    have = {c[0] for c in mcols}
    add = [c for c in ocols if c[0] not in have]
    if not add:
        return main_text
    i = main_text.rindex("]")
    entries = "".join(f"    {c!r},\n" for c in add)
    return main_text[:i].rstrip() + "\n" + entries + main_text[i:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std", default=str(HERE / "work/std"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    today = dt.date.today().isoformat()
    state = C.load_json(C.WORK / "state.json", None)
    if state is None:
        raise SystemExit("no work/finder/state.json: the find job's artifact is missing")
    entries = C.load_json(C.WORK / "new_entries.json", [])
    abstracts = C.load_json(C.WORK / "new_abstracts.json", {})
    pending = state.setdefault("pending", {})
    files = {}        # repo path -> bytes (records, skip records, converters)
    modules = {}      # repo path -> module texts sent by the standardize jobs
    outcome = []
    n_people = {}
    for d in sorted(Path(a.std).glob("*")) if Path(a.std).exists() else []:
        rp = d / "result.json"
        if not rp.exists():
            continue
        res = json.load(open(rp))
        ds = S.safe_id(res["id"])
        fd = d / "files"
        st = res["status"]
        line = {"id": ds, "status": st, "reason": res.get("reason", ""), "module": res.get("module"), "rows": res.get("rows"),
                "persons": res.get("persons"), "published_checks": res.get("published_checks"),
                "needs_person": (res.get("agent") or {}).get("needs_person", ""), "warnings": (res.get("gates") or {}).get("warnings", [])}
        if st == "released":
            problems, rec = verify_release(ds, fd)
            if problems:
                st = line["status"] = "gates_failed"
                line["reason"] = "publish check: " + "; ".join(problems)[:600]
            elif a.dry_run:
                line["url"] = "(dry run: not uploaded)"
            else:
                try:
                    res = upload(ds, fd / "source/release_extra" / f"{ds}.zip", rec)
                    rec["found_by_finder"] = today
                    line.update(url=rec["url"], version=res["version"], upload=res["status"])
                except Exception as e:  # noqa: BLE001
                    st = line["status"] = "upload_failed"
                    line["reason"] = f"{type(e).__name__}: {str(e)[:200]}"
            if st == "released":
                files[f"source/release_extra_records/{ds}.json"] = (json.dumps(rec, indent=1) + "\n").encode()
                for f in fd.rglob("*.py"):
                    rel = str(f.relative_to(fd))
                    if rel.startswith("tools/std/convert/"):
                        files[rel] = f.read_bytes()
                    elif rel.startswith("tools/std/modules/"):
                        modules.setdefault(rel, []).append(f.read_text(encoding="utf-8"))
                n_people[ds] = rec.get("persons")
                pending.pop(ds, None)
        if st in ("infra_failed", "upload_failed"):
            line["needs_person"] = line["needs_person"] or f"{'the agent could not run' if st == 'infra_failed' else 'the release upload failed'}: {line['reason']}"
            outcome.append(line)
            continue  # not counted as an attempt
        if st == "skipped":
            sk = fd / "source/release_extra_records" / f"{ds}.skip.json"
            if sk.exists():
                files[f"source/release_extra_records/{ds}.skip.json"] = sk.read_bytes()
            pending.pop(ds, None)
        elif st != "released":
            p = pending.setdefault(ds, {"listed": today, "attempts": 0})
            p["attempts"] = p.get("attempts", 0) + 1
            p["last_problem"] = (f"{st}: {line['reason']}\n\nNotes of the last attempt:\n{res.get('notes_md', '')}")[:2500]
            if res.get("review", {}).get("issues"):
                p["review"] = {"issues": res["review"]["issues"][:10]}
            final = st == "review_reject" or p["attempts"] >= S.MAX_ATTEMPTS
            if final:
                why = ("the independent review rejected it: " if st == "review_reject" else
                       f"{p['attempts']} automatic attempts did not produce a file that passed the checks; last: ")
                rec = {"id": ds, "skipped": True, "reason": "other", "details": (why + line["reason"])[:900], "sources": [],
                       "found_by_finder": today}
                files[f"source/release_extra_records/{ds}.skip.json"] = (json.dumps(rec, indent=1) + "\n").encode()
                pending.pop(ds, None)
                line["final"] = True
        outcome.append(line)
    # queued studies whose standardize job left no result (timeout or crash) count as a failed attempt
    done = {o["id"] for o in outcome}
    for ds in C.load_json(C.WORK / "queue.json", []):
        if ds in done or ds not in pending:
            continue
        p = pending[ds]
        p["attempts"] = p.get("attempts", 0) + 1
        p["last_problem"] = "the standardize job did not finish (time limit or crash); start over and finish sooner"
        line = {"id": ds, "status": "failed", "reason": "the standardize job did not finish (time limit or crash)"}
        if p["attempts"] >= S.MAX_ATTEMPTS:
            rec = {"id": ds, "skipped": True, "reason": "other", "found_by_finder": today, "sources": [],
                   "details": f"{p['attempts']} automatic attempts did not finish within the time limit."}
            files[f"source/release_extra_records/{ds}.skip.json"] = (json.dumps(rec, indent=1) + "\n").encode()
            pending.pop(ds, None)
            line["final"] = True
        outcome.append(line)
    state["history"] = (state.get("history") or [])[-51:] + [{"date": today, "listed": len(entries),
                                                               "released": sum(o["status"] == "released" for o in outcome),
                                                               "skipped": sum(o["status"] == "skipped" for o in outcome)}]
    result = {"dry_run": a.dry_run, "listed": [e["id"] for e in entries], "outcome": outcome, "pushed": False, "commit": None}
    if a.dry_run:
        C.save_json(C.WORK / "publish_result.json", result)
        S.gh_output(changed="no")
        print(f"publish (dry run): {len(entries)} would be listed, {len(outcome)} standardize results")
        return
    files["source/finder/state.json"] = (json.dumps(state, indent=1, ensure_ascii=False) + "\n").encode()
    if (C.WORK / "authors.json").exists():
        files["source/finder/authors.json"] = (C.WORK / "authors.json").read_bytes()
    msg = (f"Weekly finder {today}: list {len(entries)} new studies, standardize "
           f"{sum(o['status'] == 'released' for o in outcome)}, skip {sum(o['status'] == 'skipped' or o.get('final') for o in outcome)}")
    for attempt in range(6):
        git("fetch", "-q", "origin", "main")
        git("reset", "-q", "--hard", "origin/main")
        for k, b in files.items():
            (HERE / k).parent.mkdir(parents=True, exist_ok=True)
            (HERE / k).write_bytes(b)
        for k, texts in modules.items():  # two studies may have extended the same module: apply each in turn
            p = HERE / k
            cur = p.read_text(encoding="utf-8") if p.exists() else None
            for t in texts:
                cur = merge_module(cur, t)
            p.write_text(cur, encoding="utf-8")
        ep = HERE / "source/extra_studies.json"
        studies = json.load(open(ep, encoding="utf-8"))
        have = {e["id"] for e in studies}
        studies += [e for e in entries if e["id"] not in have]
        for e in studies:
            if e["id"] in n_people and not e.get("n_participants"):
                e["n_participants"] = n_people[e["id"]]
        ep.write_text(json.dumps(studies, indent=1, ensure_ascii=False), encoding="utf-8")
        ap_ = HERE / "source/abstracts.json"
        abs_all = json.load(open(ap_, encoding="utf-8")) if ap_.exists() else {}
        added = {k: v for k, v in abstracts.items() if k not in abs_all}
        if added:
            abs_all.update(added)
            ap_.write_text(json.dumps(abs_all, indent=1, ensure_ascii=False), encoding="utf-8")
        paths = list(files) + list(modules) + ["source/extra_studies.json"] + (["source/abstracts.json"] if added else [])
        if (HERE / "tools/explore_build.py").exists():
            before = set(git("status", "--porcelain", "--untracked-files=all").stdout.splitlines())
            r = subprocess.run([sys.executable, "-I", "tools/explore_build.py", "--incremental"], cwd=HERE, capture_output=True,
                               text=True, timeout=1800)
            print(f"explore build: exit {r.returncode}")
            after = set(git("status", "--porcelain", "--untracked-files=all").stdout.splitlines())
            for l in sorted(after - before):
                rel = l[3:].strip().strip('"')
                if rel.startswith(("source/", "site/")) and " -> " not in rel:
                    paths.append(rel)
        git("add", "--", *dict.fromkeys(paths))
        if not git("diff", "--cached", "--quiet", check=False).returncode:
            print("publish: nothing to commit")
            break
        git("-c", "user.name=OpenMotor bot", "-c", "user.email=physical.intelligence.lab@gmail.com", "commit", "-q", "-m", msg)
        if git("push", "-q", "origin", "HEAD:main", check=False).returncode == 0:
            result["pushed"] = True
            result["commit"] = git("rev-parse", "HEAD").stdout.strip()
            break
        time.sleep(15 * (attempt + 1))
    else:
        C.save_json(C.WORK / "publish_result.json", dict(result, error="could not push after 6 attempts"))
        S.gh_output(changed="no")
        raise SystemExit("could not push")
    C.save_json(C.WORK / "publish_result.json", result)
    S.gh_output(changed="yes" if result["pushed"] else "no")
    print(f"publish: {len(entries)} listed, {sum(o['status'] == 'released' for o in outcome)} standardized, pushed={result['pushed']}")


if __name__ == "__main__":
    main()
