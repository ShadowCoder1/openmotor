#!/usr/bin/env bash
# One-time setup of OpenMotor's automation (the weekly finder and the automatic standardizing of uploads).
# Run it in a normal Terminal window:   bash ~/Projects/openmotor-site/tools/setup_automation.sh
# Nothing secret is printed into any chat: tokens go from this terminal straight into GitHub and Vercel.
set -euo pipefail
REPO=physicalintelligencelab/openmotor
API_DIR="$HOME/Projects/openmotor-api"
LOG="$(mktemp -t omsetup)"
trap 'rm -f "$LOG"' EXIT

echo
echo "== Step 1 of 2: a Claude token for the automation (it runs on your Claude plan) =="
echo "A browser window will open. Log in to Claude and approve. The token is saved into the GitHub"
echo "repository's secrets; you do not need to copy it anywhere."
read -r -p "Press Enter to start (or type s and Enter to skip): " a
if [ "${a:-}" != "s" ]; then
  claude setup-token
  echo
  echo "Copy the whole token it printed (it starts with sk-ant-oat and may wrap over several lines),"
  echo "then paste it here and press Enter (it will not be shown):"
  read -r -s TOKEN
  TOKEN="$(printf '%s' "$TOKEN" | tr -d '[:space:]')"
  case "$TOKEN" in sk-ant-oat*) ;; *) echo "That does not look like a Claude token (it should start with sk-ant-oat). Nothing saved."; exit 1;; esac
  printf '%s' "$TOKEN" | gh secret set CLAUDE_CODE_OAUTH_TOKEN -R "$REPO"
  unset TOKEN
  : > "$LOG"
  echo "Saved CLAUDE_CODE_OAUTH_TOKEN in $REPO."
fi

echo
echo "== Step 2 of 2 (optional): start uploads instantly =="
echo "Without this, an upload waits up to about 15 minutes for the next scheduled check. With it, processing starts"
echo "within seconds. GitHub will open a page to create a token. On that page:"
echo "  - Resource owner: physicalintelligencelab"
echo "  - Repository access: Only select repositories -> openmotor"
echo "  - Permissions: Actions = Read and write (leave everything else as is)"
echo "  - Generate token, then copy it."
read -r -p "Press Enter to open the page (or type s and Enter to skip): " b
if [ "${b:-}" != "s" ]; then
  open "https://github.com/settings/personal-access-tokens/new?name=OpenMotor%20uploads&description=Lets%20openmotor-api%20start%20the%20intake%20workflow&target_name=physicalintelligencelab&expires_in=366&actions=write"
  echo "Paste the token here and press Enter (it will not be shown):"
  read -r -s PAT
  if [ -n "$PAT" ]; then
    (cd "$API_DIR" && printf '%s' "$PAT" | vercel env add GH_DISPATCH_TOKEN production >/dev/null && vercel deploy --prod --yes >/dev/null)
    unset PAT
    echo "Saved. The API now starts processing at once."
    echo "If GitHub says the token needs approval by the organization, approve it at"
    echo "https://github.com/organizations/physicalintelligencelab/settings/personal-access-token-requests"
  fi
fi
echo
echo "Done. You can close this window."
