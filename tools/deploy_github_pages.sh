#!/usr/bin/env bash
# Publish site changes. Since 2026-11-10 this folder is a git working copy of physicalintelligencelab/openmotor and
# GitHub Actions (.github/workflows/pages.yml) builds and deploys the site on every push. This script:
#   1. pulls the newest main (the weekly finder and the upload intake commit there on their own),
#   2. builds locally as a check,
#   3. commits the site's source paths only (the .gitignore allowlist keeps data, raw files and backups out),
#   4. pushes. The site is live about a minute later.
# usage: tools/deploy_github_pages.sh ["commit message"]     (an old first argument "owner/repo" is accepted and ignored)
set -euo pipefail
cd "$(dirname "$0")/.."
[ "${1:-}" != "" ] && [[ "$1" == */* ]] && [[ "$1" != *" "* ]] && shift
MSG="${1:-Update OpenMotor site}"
git pull --rebase --autostash -q origin main
python3 -I tools/build_page.py
git add .gitignore .github README.md source site tools
if git diff --cached --quiet; then echo "Nothing changed; nothing to deploy."; exit 0; fi
echo "Committing:"; git diff --cached --stat | tail -15
git -c user.name="Sritej Padmanabhan" -c user.email="sritej.paddy@gmail.com" commit -q -m "$MSG"
git push -q origin main
echo "Pushed. The site rebuilds in about a minute: https://github.com/physicalintelligencelab/openmotor/actions"
