#!/usr/bin/env bash
# Publish dist/ as the public OpenMotor GitHub Pages site.
#   tools/deploy_github_pages.sh <owner>/<repo> ["commit message"]      e.g. physicalintelligencelab/openmotor
# First run: creates the repo, pushes, turns on Pages. Later runs reuse .deploy (the repo's working copy),
# so only changed files are pushed. Only the site files and dataset ZIPs go up. No AI attribution in commits.
set -euo pipefail
REPO="${1:?usage: deploy_github_pages.sh owner/repo [message]}"
MSG="${2:-Update OpenMotor site}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$HERE/.deploy"
# The repo moved from ShadowCoder1/openmotor to physicalintelligencelab/openmotor on 2026-11-08, and GitHub redirects the
# old name. Resolve the name given to the repo's current name, so either name deploys to the same repo.
CANON="$(gh api "repos/$REPO" --jq .full_name 2>/dev/null || true)"
[ -n "$CANON" ] && REPO="$CANON"
URL="https://github.com/$REPO.git"
lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }
if [ -d "$STAGE/.git" ]; then
  # Reuse .deploy only if it is a copy of this same repo. Never wipe it: that would force-push a one-commit history.
  OLD="$(git -C "$STAGE" remote get-url origin 2>/dev/null || true)"
  OLDSLUG="${OLD#https://github.com/}"; OLDSLUG="${OLDSLUG%.git}"
  OLDCANON="$(gh api "repos/$OLDSLUG" --jq .full_name 2>/dev/null || true)"
  if [ "$(lower "$OLD")" = "$(lower "$URL")" ] || { [ -n "$OLDCANON" ] && [ "$(lower "$OLDCANON")" = "$(lower "$REPO")" ]; }; then
    git -C "$STAGE" remote set-url origin "$URL"
    fresh=0
  else
    echo "$STAGE holds a copy of ${OLD:-an unknown repo}, not $REPO. Nothing was changed."
    echo "To publish to a different repo, move $STAGE away yourself first."
    exit 1
  fi
else
  mkdir -p "$STAGE" && fresh=1
fi
rsync -a --delete --exclude .git --exclude .github --exclude artifact.html --exclude .DS_Store "$HERE/dist/" "$STAGE/"
find "$STAGE/data" -mindepth 1 ! -name "*.zip" -prune -exec rm -rf {} +
touch "$STAGE/.nojekyll"
cp "$HERE/source/repo_README.md" "$STAGE/README.md"
mkdir -p "$STAGE/.github" && rsync -a --delete "$HERE/source/github/" "$STAGE/.github/"
big=$(find "$STAGE" -path "$STAGE/.git" -prune -o -type f -size +95M -print)
[ -z "$big" ] || { echo "files too large for GitHub: $big"; exit 1; }
cd "$STAGE"
[ "$fresh" = 1 ] && git init -q -b main
git add -A index.html README.md catalog.csv columns.csv SHA256SUMS.txt .nojekyll data .github
if [ -d img ]; then git add -A img; else git rm -r -q --cached --ignore-unmatch img; fi
if [ -d beta ]; then git add -A beta; else git rm -r -q --cached --ignore-unmatch beta; fi  # Beta page plots (tools/beta_build.py)
if git diff --cached --quiet; then echo "Nothing changed; nothing to deploy."; exit 0; fi
git -c user.name="Sritej Padmanabhan" -c user.email="sritej.paddy@gmail.com" commit -q -m "$MSG"
if [ "$fresh" = 1 ]; then
  gh repo view "$REPO" >/dev/null 2>&1 || gh repo create "$REPO" --public --description "OpenMotor: open data for motor control and motor learning"
  git remote add origin "$URL"
  git push -u origin main --force
  gh api -X POST "repos/$REPO/pages" -f "source[branch]=main" -f "source[path]=/" >/dev/null 2>&1 || \
    gh api -X PUT "repos/$REPO/pages" -f "source[branch]=main" -f "source[path]=/" >/dev/null
  gh repo edit "$REPO" --homepage "https://${REPO%%/*}.github.io/${REPO##*/}/" >/dev/null 2>&1 || true
else
  git push origin main
fi
git --no-pager show --stat --oneline HEAD | head -12
echo "Deployed to https://${REPO%%/*}.github.io/${REPO##*/}/ (Pages rebuilds in about a minute)"
